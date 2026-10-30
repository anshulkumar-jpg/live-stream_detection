# VMS run commands (PowerShell, run from the project folder)

Project folder: `C:\Users\nj\github\vms-system`

Camera passwords: replace `PASSWORD` below, and write every `@` in a password as `%40`
(example: `Svs@2025` becomes `Svs%402025`).

---

## 1. First-time setup

```powershell
copy .env.example .env                    # skip if .env already exists
# in .env: set VMS_FERNET_KEY (44 chars, ends with "="), and every change-me-* password
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # generates a new key

docker compose --env-file .env config > $null; if ($?) { "compose-ok" }   # checks the config

New-Item -ItemType Directory -Force models | Out-Null
```

Set the key in `.env` without editing by hand:

```powershell
$k = "YOUR_FERNET_KEY"
$lines = @(Get-Content .env | Where-Object { $_ -notmatch '^\s*VMS_FERNET_KEY\s*=' })
$lines += "VMS_FERNET_KEY=$k"
Set-Content .env $lines -Encoding ascii
```

## 2. Start, stop, rebuild

```powershell
docker compose up -d                       # start everything (GPU worker, API, Kafka, MediaMTX, ...)
docker compose ps                          # what is running
docker compose down                        # stop everything

docker compose build gpu-worker-0          # rebuild the GPU worker image (large PyTorch download, 10-20 min)
docker compose up -d gpu-worker-0
docker compose up -d --build               # rebuild and start all
docker compose build --no-cache gpu-worker-0   # rebuild from scratch (redownloads PyTorch)
```

Config-only changes need no rebuild:

```powershell
docker compose restart gpu-worker-0                                        # after config\models.yaml / pipeline.yaml
docker compose up -d --force-recreate config-render mediamtx               # after config\mediamtx.yml / WEBRTC_ADDITIONAL_HOSTS
```

No-GPU test mode (fake detector, not real YOLO): put these in `.env`, then
`docker compose --profile dev up -d --build api worker-dev-0 kafka postgres mediamtx prometheus grafana`

```
VMS_REQUIRE_CUDA=false
VMS_FORCE_MOCK_MODELS=true
VMS_ALLOW_SYNTHETIC_SOURCES=true
```

## 3. Log in, then add cameras

Every API call needs a token (except `/health`, `/ready`, `/metrics`). Log in as admin first:

```powershell
$r = Invoke-RestMethod -Uri http://localhost:8000/api/v1/auth/login -Method Post -ContentType "application/json" `
     -Body (@{ username="admin"; password="YOUR_ADMIN_PASSWORD" } | ConvertTo-Json)
$h = @{ Authorization = "Bearer $($r.access_token)" }       # use -Headers $h on every call below; token lasts 60 min
```

Add `-Headers $h` to each `Invoke-RestMethod`, and `-H "Authorization: Bearer <token>"` to `curl.exe`.
The `viewer` account can list and watch but not add, change, start, stop or delete.

Dahua / CP Plus NVR: `rtsp://admin:PASSWORD@IP:554/cam/realmonitor?channel=N&subtype=1`
(`subtype=1` = lighter sub-stream, `subtype=0` = main stream).

One camera:

```powershell
$body = @{
  camera_id     = "cam_office"
  name          = "Office Floor"
  rtsp_url      = "rtsp://admin:PASSWORD@192.168.1.100:554/cam/realmonitor?channel=3&subtype=1"
  model_profile = "default"
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras -Method Post -ContentType "application/json" -Body $body
```

Several at once:

```powershell
$cams = @(
  @{ camera_id="cam_entrance"; name="Entrance";       url="rtsp://admin:PASSWORD@192.168.1.100:554/cam/realmonitor?channel=2&subtype=1" },
  @{ camera_id="cam_office";   name="Office Floor";   url="rtsp://admin:PASSWORD@192.168.1.100:554/cam/realmonitor?channel=3&subtype=1" },
  @{ camera_id="cam_store";    name="Storage Corner"; url="rtsp://admin:PASSWORD@192.168.1.100:554/cam/realmonitor?channel=4&subtype=1" },
  @{ camera_id="nvr111_ch1";   name="NVR 111 Ch1";    url="rtsp://admin:PASSWORD@192.168.1.111:554/cam/realmonitor?channel=1&subtype=1" }
)
foreach ($c in $cams) {
  $body = @{ camera_id=$c.camera_id; name=$c.name; rtsp_url=$c.url; model_profile="default" } | ConvertTo-Json
  Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras -Method Post -ContentType "application/json" -Body $body | Select-Object camera_id, desired_state, actual_state
}
```

Swagger alternative: http://localhost:8000/docs, then POST /api/v1/cameras.

## 4. Manage cameras

```powershell
curl.exe -s localhost:8000/api/v1/cameras                                   # list all
curl.exe -s localhost:8000/api/v1/cameras/cam_office                        # one camera
curl.exe -s localhost:8000/api/v1/cameras/cam_office/health                 # status, fps, errors
curl.exe -s localhost:8000/api/v1/cameras/cam_office/stream                 # stream URLs
Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras/cam_office/stop   -Method Post
Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras/cam_office/start  -Method Post
Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras/cam_office        -Method Delete

# change a camera's URL
$body = @{ rtsp_url = "rtsp://admin:PASSWORD@192.168.1.100:554/cam/realmonitor?channel=3&subtype=1" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/cameras/cam_office -Method Patch -ContentType "application/json" -Body $body
```

Delete every camera:

```powershell
(Invoke-RestMethod http://localhost:8000/api/v1/cameras).items | ForEach-Object {
  Invoke-RestMethod -Uri "http://localhost:8000/api/v1/cameras/$($_.camera_id)" -Method Delete
}
```

Detections:

```powershell
curl.exe -s "localhost:8000/api/v1/events?page_size=10"
curl.exe -s "localhost:8000/api/v1/events?page_size=10&camera_id=cam_office"
```

## 5. Where to see the output

Open in Chrome (not the VS Code built-in browser). Replace `cam_office` with the camera ID.

| What | URL |
|---|---|
| Live video with boxes (WebRTC) | http://localhost:8889/cam_office/ |
| API docs | http://localhost:8000/docs |
| Grafana dashboards | http://localhost:3000 |
| Prometheus | http://127.0.0.1:9090 |

No video is recorded or saved: the live streams (HLS and WebRTC) exist only in MediaMTX's memory.

## 6. Troubleshooting

```powershell
docker compose ps
docker compose logs gpu-worker-0 --tail 50
docker compose logs mediamtx --tail 30
docker compose logs api --tail 30

docker compose exec gpu-worker-0 python /app/scripts/test_gpu.py            # is CUDA visible in the container?
Test-NetConnection 192.168.1.100 -Port 554                                  # can this PC reach the NVR?
```

| Symptom | Cause / fix |
|---|---|
| Camera `pending`, "no workers registered" | The worker is not running: `docker compose up -d gpu-worker-0` |
| Worker exits: "weights file not found" | Put `best.pt` in `models\` |
| Worker: "no kernel image" (RTX 50-series) | Needs the cu128 PyTorch image: rebuild `gpu-worker-0` |
| Camera `reconnecting`, "Connection timed out" | PC and NVR are on different networks, or the IP is wrong |
| `404 Not Found` | Wrong RTSP path for the camera brand |
| `401 Unauthorized` | Wrong NVR password (check the NVR web login first), or the account is locked, wait 30 min |
| `Invalid data found` | The NVR channel has no camera attached |
| WebRTC page spinner | Use Chrome; make sure `WEBRTC_ADDITIONAL_HOSTS` includes `127.0.0.1` |
| 409 "already exists" when adding | The camera is already added |

## 7. Notes

- Watching a stream needs the token too: open `http://localhost:8889/<camera_id>/`, and when the browser asks for a username and password, enter any username and paste the token as the password.
- Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md).
