"""HLS quality ladder: master playlist (adaptive bitrate) and a small browser viewer with a quality menu."""
from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse

from app.api.deps import User, get_container, user_bearer_or_query
from app.core.config import load_pipeline_config
from app.core.security import ValidationError, validate_camera_id
from app.db.repositories import CameraRepo
from app.services.container import Container

router = APIRouter(tags=["viewer"])


def rendition_path(camera_id: str, height: int, main_height: int) -> str:
    return camera_id if height == main_height else f"{camera_id}~{height}p"


@router.get("/api/v1/streams/{camera_id}/master.m3u8")
async def master_playlist(camera_id: str, request: Request, token: str | None = Query(None), c: Container = Depends(get_container),
                          _: User = Depends(user_bearer_or_query)):
    """Adaptive-bitrate master playlist. Players pick the quality automatically or the user pins one."""
    try:
        validate_camera_id(camera_id)
    except ValidationError as e:
        raise HTTPException(400, str(e))
    async with c.sf() as db:
        if await CameraRepo(db).get(camera_id) is None:
            raise HTTPException(404, f"camera '{camera_id}' not found")
    enc = load_pipeline_config(c.s.pipeline_config).encoder
    pub = urlsplit(c.s.public_hls_base)  # same host the client used to reach the API, so LAN and localhost both work
    base = f"{pub.scheme}://{request.url.hostname}:{pub.port or 8888}"
    q = f"?token={token}" if token else ""
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for h in enc.heights():
        bw = int(enc.rendition_bitrate(h) * 1000 * 1.1)
        lines.append(f"#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={enc.rendition_width(h)}x{h},FRAME-RATE={enc.fps}")
        lines.append(f"{base}/{rendition_path(camera_id, h, enc.height)}/index.m3u8{q}")
    return Response("\n".join(lines) + "\n", media_type="application/vnd.apple.mpegurl",
                    headers={"Cache-Control": "no-store"})


@router.get("/viewer/{camera_id}", response_class=HTMLResponse, include_in_schema=False)
async def viewer(camera_id: str, c: Container = Depends(get_container)):
    try:
        validate_camera_id(camera_id)
    except ValidationError:
        raise HTTPException(400, "invalid camera id")
    return HTMLResponse(_PAGE.replace("__CAMERA__", camera_id).replace("__AUTH__", "true" if c.s.auth_enabled else "false"))


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live - __CAMERA__</title>
<style>
 [hidden]{display:none!important}
 body{margin:0;background:#0d0f12;color:#e6e8eb;font:14px system-ui,sans-serif}
 header{display:flex;gap:12px;align-items:center;padding:10px 16px;background:#161a20}
 header b{flex:1}
 select,input,button{background:#222831;color:inherit;border:1px solid #333b47;border-radius:6px;padding:6px 10px;font:inherit}
 video{display:block;width:100%;max-height:calc(100vh - 52px);background:#000}
 #msg{padding:8px 16px;color:#9aa4b2}
 #login{max-width:280px;margin:80px auto;display:flex;flex-direction:column;gap:10px}
</style></head><body>
<div id="login" hidden>
 <b>Sign in</b>
 <input id="u" placeholder="username" autocomplete="username">
 <input id="p" type="password" placeholder="password" autocomplete="current-password">
 <button id="go">Sign in</button><span id="lerr"></span>
</div>
<div id="app" hidden>
 <header><b>__CAMERA__</b>
  <label>Quality <select id="q"><option value="-1">Auto</option></select></label>
  <span id="cur"></span></header>
 <div id="msg"></div>
 <video id="v" controls autoplay muted playsinline></video>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.17/hls.min.js"></script>
<script>
const CAM="__CAMERA__",AUTH=__AUTH__;
let token=new URLSearchParams(location.search).get("token");
try{token=token||sessionStorage.getItem("vms_token")}catch(e){}
const $=id=>document.getElementById(id);
function start(){
 $("login").hidden=true;$("app").hidden=false;
 const src=`/api/v1/streams/${CAM}/master.m3u8`+(AUTH?`?token=${encodeURIComponent(token)}`:""),v=$("v"),sel=$("q");
 if(window.Hls&&Hls.isSupported()){
  const hls=new Hls({lowLatencyMode:false});
  hls.loadSource(src);hls.attachMedia(v);
  hls.on(Hls.Events.MANIFEST_PARSED,()=>{
   $("msg").textContent="";
   hls.levels.map((l,i)=>({i,h:l.height})).sort((a,b)=>b.h-a.h).forEach(o=>{
    const e=document.createElement("option");e.value=o.i;e.textContent=o.h+"p";sel.appendChild(e)});
  });
  hls.on(Hls.Events.LEVEL_SWITCHED,(_,d)=>{const l=hls.levels[d.level];$("cur").textContent=l?"playing "+l.height+"p":""});
  hls.on(Hls.Events.ERROR,(_,d)=>{
   if(d.fatal){$("msg").textContent="Stream not available yet, retrying...";
    setTimeout(()=>{hls.loadSource(src)},3000);}});
  sel.onchange=()=>{hls.currentLevel=parseInt(sel.value)};
 }else if(v.canPlayType("application/vnd.apple.mpegurl")){
  v.src=src;sel.disabled=true;$("msg").textContent="Native HLS player: automatic quality only.";
 }else $("msg").textContent="This browser cannot play HLS.";
}
if(token||!AUTH){start()}else{
 $("login").hidden=false;
 $("go").onclick=async()=>{
  const r=await fetch("/api/v1/auth/login",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({username:$("u").value,password:$("p").value})});
  if(!r.ok){$("lerr").textContent="Invalid credentials";return}
  token=(await r.json()).access_token;try{sessionStorage.setItem("vms_token",token)}catch(e){}start();};
}
</script></body></html>"""
