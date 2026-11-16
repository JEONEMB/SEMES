// 3D 화면 조작 (도면과 같음: 왼쪽 드래그 회전, 오른쪽 드래그 이동, 휠 확대) + SEMES 화각 카메라 화면
using UnityEngine;
using UnityEngine.EventSystems;

namespace YT
{
    public class OrbitCam : MonoBehaviour
    {
        public Vector3 target = Vector3.zero;
        public float minDist = 60, maxDist = 3200;
        public System.Func<bool> pointerBlocked;
        Vector3 twP0, twP1, twT0, twT1; float twStart = -1;
        Vector3 lastMouse; int dragButton = -1;

        public void Set(Vector3 pos, Vector3 tgt) { transform.position = pos; target = tgt; transform.LookAt(target); twStart = -1; }

        public void TweenTo(Vector3 pos, Vector3 tgt)
        {
            twP0 = transform.position; twT0 = target; twP1 = pos; twT1 = tgt; twStart = Time.unscaledTime;
        }

        bool Blocked() => (pointerBlocked != null && pointerBlocked()) || (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject());

        void LateUpdate()
        {
            if (twStart >= 0)
            {
                float k = Mathf.Min(1, (Time.unscaledTime - twStart) / 0.7f), e = k < .5f ? 2 * k * k : 1 - Mathf.Pow(-2 * k + 2, 2) / 2;
                transform.position = Vector3.Lerp(twP0, twP1, e); target = Vector3.Lerp(twT0, twT1, e);
                transform.LookAt(target);
                if (k >= 1) twStart = -1;
                return;
            }
            var m = Input.mousePosition;
            if (dragButton < 0)
            {
                for (int b = 0; b < 3; b++) if (Input.GetMouseButtonDown(b) && !Blocked()) { dragButton = b; lastMouse = m; }
            }
            else if (!Input.GetMouseButton(dragButton)) dragButton = -1;
            if (dragButton >= 0)
            {
                Vector3 d = m - lastMouse; lastMouse = m;
                if (dragButton == 0)
                {
                    var off = transform.position - target;
                    off = Quaternion.AngleAxis(d.x * 0.3f, Vector3.up) * off;
                    var right = Vector3.Cross(Vector3.up, off).normalized;
                    var o2 = Quaternion.AngleAxis(-d.y * 0.3f, right) * off;
                    if (Vector3.Angle(o2, Vector3.up) > 2 && Vector3.Angle(o2, Vector3.down) > 2) off = o2;
                    transform.position = target + off;
                }
                else
                {
                    float dist = (transform.position - target).magnitude;
                    float s = dist * Mathf.Tan(GetComponent<Camera>().fieldOfView * 0.5f * Mathf.Deg2Rad) * 2 / Mathf.Max(1, GetComponent<Camera>().pixelHeight);
                    var pan = (-transform.right * d.x - transform.up * d.y) * s;
                    transform.position += pan; target += pan;
                }
                transform.LookAt(target);
            }
            float w = Input.mouseScrollDelta.y;
            if (Mathf.Abs(w) > 0.01f && !Blocked())
            {
                var off = transform.position - target;
                float dist = Mathf.Clamp(off.magnitude * Mathf.Pow(0.9f, w), minDist, maxDist);
                transform.position = target + off.normalized * dist;
            }
        }
    }

    // SEMES 카메라 화각 = GitLab dev verify_web/geometry.py 알고리즘 입력 행렬 (실제 카메라 사진을 이 값으로 다시 투영해 알고리즘에 넣음)
    //   fx = 12/2.8µm × 3840/4096 = 4017.86 (가로만 15/16), fy = 12/2.8µm = 4285.71, 광학 중심 = 사진 가운데, 3840×2160
    //   (이전: zip 도면 fx 4450, fy 4450×16/15 — 2026-11-15 verify_web 값으로 바꿈)
    //   광학 중심 위치: 렌즈 앞면 뒤 2.2 (qr_target make_target.py PUPIL, 'fx 4450 기준 추정' 값 그대로 — verify_web 값 기준 근거는 없음)
    public class SemesView
    {
        public const float FX = 12f / 2.8e-3f * 3840f / 4096f, FY = 12f / 2.8e-3f, W = 3840, H = 2160;
        public Camera cam;
        public RenderTexture rt;
        public Texture2D overlay;

        public static SemesView Make(Transform mount)
        {
            var v = new SemesView();
            var go = new GameObject("SemesCamera");
            go.transform.SetParent(mount, false);
            v.cam = go.AddComponent<Camera>();
            v.cam.nearClipPlane = 5; v.cam.farClipPlane = 3000;
            v.cam.clearFlags = CameraClearFlags.SolidColor;
            v.cam.backgroundColor = new Color(0.93f, 0.94f, 0.92f);
            v.rt = new RenderTexture(1920, 1080, 24) { name = "SemesView", antiAliasing = 4 };
            v.cam.targetTexture = v.rt;
            v.cam.fieldOfView = 2 * Mathf.Atan(1080f / FY) * Mathf.Rad2Deg;
            v.cam.aspect = (1920f / FX) / (1080f / FY);
            v.overlay = MakeOverlay();
            return v;
        }

        // 화면 좌표 (3840×2160, 가운데 0, 오른쪽·위 +) — 도면 readQR 과 같은 정의
        public Vector2 Px(Vector3 world)
        {
            cam.aspect = (1920f / FX) / (1080f / FY);
            var vp = cam.WorldToViewportPoint(world);
            return new Vector2((vp.x - 0.5f) * W, (vp.y - 0.5f) * H);
        }

        // SEMES 사진의 특징 위치 (도면 insetRef: QR 3개, 나사 4개, 가로선 4개). 960×540 (1/4) 투명 그림
        static Texture2D MakeOverlay()
        {
            const int w = 960, h = 540; const float s = 0.25f;
            var t = new Texture2D(w, h, TextureFormat.RGBA32, false) { name = "SemesRef", wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[w * h];
            var red = new Color32(224, 48, 42, 255);
            void Dot(int x, int y) { if (x >= 0 && x < w && y >= 0 && y < h) px[(h - 1 - y) * w + x] = red; }
            void Thick(float x, float y) { for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) Dot(Mathf.RoundToInt(x) + a, Mathf.RoundToInt(y) + b); }
            void Line(float x0, float y0, float x1, float y1)
            {
                int n = Mathf.CeilToInt(Mathf.Max(Mathf.Abs(x1 - x0), Mathf.Abs(y1 - y0)) * s) + 1;
                for (int i = 0; i <= n; i++) { float k = i / (float)n; Thick((x0 + (x1 - x0) * k) * s, (y0 + (y1 - y0) * k) * s); }
            }
            void Rect(float x, float y, float rw, float rh) { Line(x, y, x + rw, y); Line(x + rw, y, x + rw, y + rh); Line(x + rw, y + rh, x, y + rh); Line(x, y + rh, x, y); }
            void Circle(float cx, float cy, float r) { for (int i = 0; i < 90; i++) { float a = i / 90f * Mathf.PI * 2; Thick((cx + r * Mathf.Cos(a)) * s, (cy + r * Mathf.Sin(a)) * s); } }
            Rect(1557, 965, 186, 200); Rect(1814, 965, 186, 200); Rect(2072, 965, 186, 200);
            Circle(960, 1352, 34); Circle(2867, 1357, 34); Circle(1010, 1677, 34); Circle(2820, 1684, 34);
            Line(400, 716, 1168, 716); Line(2700, 723, 3468, 723); Line(400, 1788, 1168, 1788); Line(2600, 1797, 3368, 1797);
            t.SetPixels32(px); t.Apply();
            return t;
        }
    }
}
