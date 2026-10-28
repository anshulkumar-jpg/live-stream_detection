#!/usr/bin/env python3
"""Generate Grafana dashboard JSON (provisioned from monitoring/grafana/dashboards)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}


def panel(i, title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, legend="{{worker_id}} {{camera_id}}"):
    p = {"id": i, "type": kind, "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + n), "expr": e, "legendFormat": legend, "datasource": DS} for n, e in enumerate(exprs)],
         "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": []}}
    return p


def dash(uid, title, panels, variables=()):
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "refresh": "10s", "time": {"from": "now-1h", "to": "now"},
            "tags": ["vms"], "panels": panels,
            "templating": {"list": [{"name": v, "type": "query", "datasource": DS, "query": f"label_values({m}, {v})",
                                     "refresh": 2, "includeAll": True, "multi": True, "current": {"text": "All", "value": "$__all"}}
                                    for v, m in variables]}}


def lat(q, metric, by="worker_id"):
    return f"histogram_quantile({q}, sum by (le, {by}) (rate({metric}_bucket[5m])))"


D = {}
D["system-overview"] = dash("vms-overview", "VMS - System overview", [
    panel(1, "Healthy workers (heartbeat < 15s)", ["count(worker_heartbeat_age_seconds < 15) or vector(0)"], 0, 0, 4, 4, "stat", legend=""),
    panel(2, "Active cameras (sum over workers)", ["sum(active_cameras) or vector(0)"], 4, 0, 4, 4, "stat", legend=""),
    panel(3, "Connected cameras", ["sum(camera_connection_status) or vector(0)"], 8, 0, 4, 4, "stat", legend=""),
    panel(4, "Dropped frames / s (all)", ["sum(rate(dropped_frames_total[5m]))"], 12, 0, 6, 4, "stat", legend=""),
    panel(5, "Detections / s (all)", ["sum(rate(detections_total[1m]))"], 18, 0, 6, 4, "stat", legend=""),
    panel(6, "Worker heartbeat age", ["worker_heartbeat_age_seconds"], 0, 4, 12, 8, unit="s", legend="{{worker_id}}"),
    panel(7, "Active cameras per worker", ["active_cameras"], 12, 4, 12, 8, legend="{{worker_id}}"),
    panel(8, "End-to-end latency P95", [lat(0.95, "end_to_end_latency_seconds")], 0, 12, 12, 8, unit="s", legend="{{worker_id}}"),
    panel(9, "Stream publish status", ["stream_publish_status"], 12, 12, 12, 8),
])
D["camera-health"] = dash("vms-camera", "VMS - Per-camera health", [
    panel(1, "Connection status", ["camera_connection_status{camera_id=~\"$camera_id\"}"], 0, 0, 24, 6, "state-timeline"),
    panel(2, "Decoder FPS", ["decoder_fps{camera_id=~\"$camera_id\"}"], 0, 6),
    panel(3, "Inference FPS", ["inference_fps{camera_id=~\"$camera_id\"}"], 12, 6),
    panel(4, "Output FPS", ["output_fps{camera_id=~\"$camera_id\"}"], 0, 14),
    panel(5, "Queue depth", ["frame_queue_depth{camera_id=~\"$camera_id\"}"], 12, 14),
    panel(6, "Dropped frames / s by reason", ["sum by (camera_id, reason) (rate(dropped_frames_total{camera_id=~\"$camera_id\"}[5m]))"], 0, 22, legend="{{camera_id}} {{reason}}"),
    panel(7, "FFmpeg restarts (increase 1h)", ["sum by (camera_id, role) (increase(ffmpeg_restart_total{camera_id=~\"$camera_id\"}[1h]))"], 12, 22, legend="{{camera_id}} {{role}}"),
    panel(8, "End-to-end latency P95", [lat(0.95, "end_to_end_latency_seconds{camera_id=~\"$camera_id\"}", "camera_id").replace("_seconds{camera_id=~\"$camera_id\"}_bucket", "_seconds_bucket{camera_id=~\"$camera_id\"}")], 0, 30, unit="s", legend="{{camera_id}}"),
    panel(9, "Detections / s by class", ["sum by (camera_id, class_name) (rate(detections_total{camera_id=~\"$camera_id\"}[1m]))"], 12, 30, legend="{{camera_id}} {{class_name}}"),
], [("camera_id", "camera_connection_status")])
D["gpu-workers"] = dash("vms-gpu", "VMS - GPU worker utilization", [
    panel(1, "GPU utilization %", ["gpu_utilization_percent{worker_id=~\"$worker_id\"}"], 0, 0, unit="percent", legend="{{worker_id}} gpu{{gpu_id}}"),
    panel(2, "GPU memory used", ["gpu_memory_used_bytes{worker_id=~\"$worker_id\"}"], 12, 0, unit="bytes", legend="{{worker_id}} gpu{{gpu_id}}"),
    panel(3, "GPU memory fraction", ["gpu_memory_used_bytes / gpu_memory_total_bytes"], 0, 8, unit="percentunit", legend="{{worker_id}}"),
    panel(4, "Cameras per worker", ["active_cameras{worker_id=~\"$worker_id\"}"], 12, 8, legend="{{worker_id}}"),
    panel(5, "Total queue depth per worker", ["sum by (worker_id) (frame_queue_depth{worker_id=~\"$worker_id\"})"], 0, 16, legend="{{worker_id}}"),
    panel(6, "Models ready", ["model_ready{worker_id=~\"$worker_id\"}"], 12, 16, kind="state-timeline", legend="{{worker_id}} {{model}}"),
], [("worker_id", "worker_heartbeat_age_seconds")])
D["inference"] = dash("vms-inference", "VMS - Inference latency & throughput", [
    panel(1, "Model inference latency P50/P95/P99", [lat(0.5, "inference_latency_seconds", "worker_id, model"), lat(0.95, "inference_latency_seconds", "worker_id, model"), lat(0.99, "inference_latency_seconds", "worker_id, model")], 0, 0, 12, 8, unit="s", legend="{{worker_id}} {{model}}"),
    panel(2, "End-to-end latency P50/P95/P99", [lat(0.5, "end_to_end_latency_seconds"), lat(0.95, "end_to_end_latency_seconds"), lat(0.99, "end_to_end_latency_seconds")], 12, 0, 12, 8, unit="s", legend="{{worker_id}}"),
    panel(3, "Inference FPS per worker (sum of cameras)", ["sum by (worker_id) (inference_fps)"], 0, 8, legend="{{worker_id}}"),
    panel(4, "Decoder FPS per worker", ["sum by (worker_id) (decoder_fps)"], 12, 8, legend="{{worker_id}}"),
    panel(5, "Output FPS per worker", ["sum by (worker_id) (output_fps)"], 0, 16, legend="{{worker_id}}"),
    panel(6, "Dropped frames / s by reason", ["sum by (worker_id, reason) (rate(dropped_frames_total[5m]))"], 12, 16, legend="{{worker_id}} {{reason}}"),
])
D["kafka"] = dash("vms-kafka", "VMS - Kafka health", [
    panel(1, "Publish errors / drops (increase 5m)", ["sum by (component, reason) (increase(kafka_publish_errors_total[5m]))"], 0, 0, legend="{{component}} {{reason}}"),
    panel(2, "Control-plane commands by result", ["sum by (type, result) (rate(control_plane_commands_total[5m]))"], 12, 0, legend="{{type}} {{result}}"),
    panel(3, "Broker up (kafka-exporter)", ["kafka_brokers"], 0, 8, 6, 6, "stat", legend=""),
    panel(4, "Consumer group lag", ["sum by (consumergroup, topic) (kafka_consumergroup_lag)"], 6, 8, 18, 8, legend="{{consumergroup}} {{topic}}"),
    panel(5, "Messages in / s by topic", ["sum by (topic) (rate(kafka_topic_partition_current_offset[1m]))"], 0, 16, 24, 8, legend="{{topic}}"),
])
D["mediamtx"] = dash("vms-mediamtx", "VMS - MediaMTX streaming health", [
    panel(1, "Encoder publishing (1=yes)", ["stream_publish_status"], 0, 0, kind="state-timeline"),
    panel(2, "MediaMTX scrape up", ["up{job=\"mediamtx\"}"], 12, 0, 6, 8, "stat", legend=""),
    panel(3, "Paths", ["paths"], 18, 0, 6, 8, "stat", legend="{{state}}"),
    panel(4, "RTSP sessions", ["rtsp_conns", "rtsp_sessions"], 0, 8, legend="{{__name__}} {{state}}"),
    panel(5, "WebRTC sessions", ["webrtc_sessions"], 12, 8, legend="{{state}}"),
    panel(6, "HLS muxers", ["hls_muxers"], 0, 16, legend="{{name}}"),
    panel(7, "Bytes sent / s (WebRTC, HLS)", ["rate(webrtc_sessions_bytes_sent[1m])", "rate(hls_muxers_bytes_sent[1m])"], 12, 16, unit="Bps", legend="{{__name__}}"),
    panel(8, "Encoder output FPS", ["output_fps"], 0, 24, 24, 8),
])
OUT.mkdir(parents=True, exist_ok=True)
for name, d in D.items():
    (OUT / f"{name}.json").write_text(json.dumps(d, indent=1))
print("wrote", len(D), "dashboards to", OUT)
