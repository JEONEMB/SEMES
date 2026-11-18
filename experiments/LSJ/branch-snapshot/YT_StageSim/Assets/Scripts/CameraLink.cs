// 카메라 출처: 시뮬레이터 가상 카메라 / 라즈베리파이 영상 (PiBridge/pi_bridge.py, verify_web 카메라·변환 코드 그대로 사용)
// 알고리즘 입력은 둘 다 3840×2160 8bit 흑백 (verify_web Rectifier 와 같은 크기)
// 상태: 파이 쪽은 아직 실제로 받아 본 적 없음 (월요일 현장 첫 시험)
using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace YT
{
    public interface ICam
    {
        string Kind { get; }            // sim / pi
        string Label { get; }
        bool Live { get; }
        Texture View { get; }           // 화면에 띄울 그림
        float Fps { get; }
        Task<(bool settled, double seconds, double shiftPx)> Settle(double px, double timeout, Func<bool> aborted);
        Task<(byte[] gray, int w, int h)> Grab(int average);
        void SetView(int zoom, bool plate);
        void Tick();
        void Close();
    }

    // 시뮬레이터: SEMES 화각 가상 카메라 (흔들림 없음 → 이동 끝나면 안정)
    public class SimCam : ICam
    {
        readonly App app;
        public SimCam(App app) { this.app = app; }
        public string Kind => "sim";
        public string Label => "시뮬레이터 가상 카메라";
        public bool Live => true;
        public Texture View => app.semes.rt;
        public float Fps => 1f / Mathf.Max(1e-3f, Time.smoothDeltaTime);

        public async Task<(bool, double, double)> Settle(double px, double timeout, Func<bool> aborted)
        {
            float t0 = Time.realtimeSinceStartup;
            while (app.stage.Status.state != StState.Idle && Time.realtimeSinceStartup - t0 < timeout) { if (aborted()) throw new Aborted(); await Awaitable.NextFrameAsync(); }
            await Awaitable.NextFrameAsync();
            return (app.stage.Status.state == StState.Idle, Time.realtimeSinceStartup - t0, 0.0);
        }

        public Task<(byte[], int, int)> Grab(int average)
        {
            const int W = 3840, H = 2160;
            var cam = app.semes.cam;
            var rt = RenderTexture.GetTemporary(W, H, 24); rt.antiAliasing = 1;
            var old = cam.targetTexture; var oldActive = RenderTexture.active;
            cam.targetTexture = rt; cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0); tex.Apply(false);
            cam.targetTexture = old; RenderTexture.active = oldActive; RenderTexture.ReleaseTemporary(rt);
            var px = tex.GetPixels32(); UnityEngine.Object.Destroy(tex);
            var g = new byte[W * H];
            for (int y = 0; y < H; y++)
            {
                int src = (H - 1 - y) * W, dst = y * W;
                for (int x = 0; x < W; x++) { var c = px[src + x]; g[dst + x] = (byte)((c.r * 299 + c.g * 587 + c.b * 114) / 1000); }
            }
            return Task.FromResult((g, W, H));
        }

        public void SetView(int zoom, bool plate) { }
        public void Tick() { }
        public void Close() { }
    }

    // 라즈베리파이 pi_bridge.py
    public class PiCam : ICam
    {
        readonly string baseUrl;
        readonly HttpClient http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        // 알고리즘용 사진(/frame, 3840×2160 흑백 약 8.3 MB)만 따로 120초. 와이파이가 붐빌 때 60·89초 걸린 기록이 있어 (11-16 현장) 30초로는 끊김
        // 나머지 요청(/health·/settle·/view·/video)은 작아서 30초 그대로 (늘리면 파이가 꺼졌을 때 연결 버튼에서 오래 멈춤)
        public const int FRAME_TIMEOUT_S = 120;
        readonly HttpClient frameHttp = new HttpClient { Timeout = TimeSpan.FromSeconds(FRAME_TIMEOUT_S) };
        readonly Texture2D tex = new Texture2D(1280, 720, TextureFormat.RGB24, false);
        byte[] latest; int latestSeq, shownSeq; readonly object gate = new object();
        Thread mjpeg; volatile bool closed; volatile string error;
        float fps; double lastFrameAt;
        public float[] Plate = { 640, 360 };
        // 알고리즘 좌표 → 화면(1280×720) 좌표: 화면 x = a·x + b, 화면 y = c·y + d (verify_web ViewMapper.to_view 는 축마다 1차식). 옛 pi_bridge 는 안 줌 → null
        public float[] ViewMap;

        [Serializable] class Health { public bool ok; public float fps; public string error; public string kind; public float[] plate; public float[] view_map; }

        public PiCam(string url)
        {
            baseUrl = url.TrimEnd('/');
            var h = HealthNow();
            if (!h.ok) throw new RunError("파이 카메라 응답은 왔지만 영상이 없습니다: " + (h.error ?? "?"));
            if (h.plate != null && h.plate.Length == 2) Plate = h.plate;
            if (h.view_map != null && h.view_map.Length == 4) ViewMap = h.view_map;
            mjpeg = new Thread(ReadVideo) { IsBackground = true, Name = "pi-mjpeg" }; mjpeg.Start();
        }

        Health HealthNow()
        {
            try { return JsonUtility.FromJson<Health>(http.GetStringAsync(baseUrl + "/health").Result); }
            catch (Exception e) { throw new RunError($"파이 카메라 {baseUrl} 에 연결할 수 없습니다: {(e.InnerException ?? e).Message}"); }
        }

        public string Kind => "pi";
        public string Label => "라즈베리파이 영상 · " + baseUrl;
        public bool Live => error == null && !closed && Time.realtimeSinceStartupAsDouble - lastFrameAt < 3;   // verify_web·demo_web: 3초 넘게 프레임이 없으면 끊김
        public Texture View => tex;
        public float Fps => fps;
        public string Error => error;

        // multipart/x-mixed-replace (boundary=frame) 에서 JPEG 만 꺼냄
        void ReadVideo()
        {
            while (!closed)
            {
                try
                {
                    using (var resp = http.GetAsync(baseUrl + "/video", HttpCompletionOption.ResponseHeadersRead).Result)
                    using (var s = resp.Content.ReadAsStreamAsync().Result)
                    {
                        error = null;
                        while (!closed)
                        {
                            string line;
                            do line = ReadLine(s); while (line != null && !line.StartsWith("--"));   // 경계 "--frame"
                            if (line == null) break;
                            int len = -1;
                            while ((line = ReadLine(s)) != null && line.Length > 0)
                                if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase)) len = int.Parse(line.Substring(15).Trim());
                            if (line == null || len < 0) break;
                            var buf = new byte[len]; int got = 0;
                            while (got < len) { int n = s.Read(buf, got, len - got); if (n <= 0) throw new IOException("영상 끊김"); got += n; }
                            lock (gate) { latest = buf; latestSeq++; }
                        }
                    }
                }
                catch (Exception e) { if (!closed) error = "파이 영상 끊김: " + (e.InnerException ?? e).Message; Thread.Sleep(3000); }
            }
        }

        static string ReadLine(Stream s)
        {
            var sb = new StringBuilder(); int b;
            while ((b = s.ReadByte()) >= 0) { if (b == '\n') return sb.ToString().TrimEnd('\r'); sb.Append((char)b); if (sb.Length > 1024) sb.Clear(); }
            return null;
        }

        public void Tick()
        {
            byte[] jpg = null;
            lock (gate) if (latestSeq != shownSeq) { jpg = latest; shownSeq = latestSeq; }
            if (jpg == null) return;
            tex.LoadImage(jpg, false);
            double now = Time.realtimeSinceStartupAsDouble;
            if (lastFrameAt > 0) fps = Mathf.Lerp(fps <= 0 ? (float)(1 / (now - lastFrameAt)) : fps, (float)(1 / Math.Max(1e-3, now - lastFrameAt)), 0.2f);
            lastFrameAt = now;
        }

        [Serializable] class SettleR { public bool settled; public double seconds; public double shift_px; }

        public async Task<(bool, double, double)> Settle(double px, double timeout, Func<bool> aborted)
        {
            var s = await http.GetStringAsync($"{baseUrl}/settle?px={px.ToString(System.Globalization.CultureInfo.InvariantCulture)}&timeout={timeout.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            var r = JsonUtility.FromJson<SettleR>(s);
            return (r.settled, r.seconds, r.shift_px);
        }

        public double LastGrabS;   // 마지막 사진 받기에 걸린 시간 (초)

        public async Task<(byte[], int, int)> Grab(int average)
        {
            var t0 = DateTime.Now;
            try
            {
                using (var resp = await frameHttp.GetAsync($"{baseUrl}/frame?average={average}"))
                {
                    if (!resp.IsSuccessStatusCode) throw new RunError("파이 사진 받기 실패: " + await resp.Content.ReadAsStringAsync());
                    var b = await resp.Content.ReadAsByteArrayAsync();
                    int w = int.Parse(string.Join("", resp.Headers.GetValues("X-Width"))), h = int.Parse(string.Join("", resp.Headers.GetValues("X-Height")));
                    if (b.Length != w * h) throw new RunError($"파이 사진 크기가 맞지 않음 ({b.Length} ≠ {w}×{h})");
                    LastGrabS = (DateTime.Now - t0).TotalSeconds;
                    return (b, w, h);
                }
            }
            catch (TaskCanceledException) when (!closed)
            {   // HttpClient 시간 초과는 "A task was canceled." 로만 나와서 이유를 적어 줌
                throw new RunError($"파이 사진 받기 시간 초과 ({FRAME_TIMEOUT_S}초, 3840×2160 흑백 약 8.3 MB). 와이파이가 느린지 확인하세요.");
            }
        }

        public void SetView(int zoom, bool plate)
        {
            var body = new StringContent($"{{\"zoom\": {zoom}, \"center\": \"{(plate ? "plate" : "image")}\"}}", Encoding.UTF8, "application/json");
            _ = Task.Run(async () =>
            {   // 파이가 확대를 바꾼 뒤의 표시 위치를 받음
                try
                {
                    (await http.PostAsync(baseUrl + "/view", body)).Dispose();
                    var h = HealthNow();
                    if (h.plate != null && h.plate.Length == 2) Plate = h.plate;
                    if (h.view_map != null && h.view_map.Length == 4) ViewMap = h.view_map;
                }
                catch { }
            });
        }

        public void Close() { closed = true; try { http.CancelPendingRequests(); http.Dispose(); frameHttp.CancelPendingRequests(); frameHttp.Dispose(); } catch { } }
    }
}
