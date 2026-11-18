// 웹 demo_web/static/style.css 와 같은 모양을 uGUI 로 만드는 도우미 (색·여백·버튼 종류·패널 제목 줄)
// 크기: 웹 px × 약 1.2 (웹은 1440 폭 기준, Unity 화면 기준은 1920×1080)
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace YT
{
    public static class W
    {
        // 웹 :root 색
        public static readonly Color BG = Hex(0xffffff), SURFACE2 = Hex(0xf4f7fc), LINE = Hex(0xdce3ed), INK = Hex(0x243349), MUTED = Hex(0x637187),
            ACCENT = Hex(0x0A3D91), RED = Hex(0xdf3544), STOP = Hex(0xb12c39), AMBER = Hex(0x98600d), HEAD = Hex(0xf7f9fc),
            PILL_BG = Hex(0xeef4fd), PILL_LINE = Hex(0xc5d5ef), BTN_LINE = Hex(0xbacbe4), INPUT_LINE = Hex(0xc8d2e1), TRACK = Hex(0xe3eaf4),
            END_BG = Hex(0xeef4fd), WARN_BG = Hex(0xfff6e8), WARN_INK = Hex(0x8c5509), ROW_LINE = Hex(0xe3e8f0);
        public static Color Hex(uint c) => new Color(((c >> 16) & 255) / 255f, ((c >> 8) & 255) / 255f, (c & 255) / 255f);

        static Sprite round, ring, dot;
        public static Font Mono;

        // 둥근 모서리 9-slice (반지름 6, 웹 --radius)
        static Sprite MakeRound(bool fill)
        {
            const int n = 32, r = 7;
            var t = new Texture2D(n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
                {
                    float cx = Mathf.Clamp(x + .5f, r, n - r), cy = Mathf.Clamp(y + .5f, r, n - r);
                    float d = Vector2.Distance(new Vector2(x + .5f, y + .5f), new Vector2(cx, cy));
                    float a = fill ? Mathf.Clamp01(r - d + .5f) : Mathf.Clamp01(r - d + .5f) * Mathf.Clamp01(d - (r - 1.3f) + .5f);
                    if (!fill && (x < 1 || y < 1 || x >= n - 1 || y >= n - 1)) a = Mathf.Max(a, (x > r && x < n - r) || (y > r && y < n - r) ? 1 : a);
                    t.SetPixel(x, y, new Color(1, 1, 1, a));
                }
            t.Apply();
            return Sprite.Create(t, new Rect(0, 0, n, n), new Vector2(.5f, .5f), 100, 0, SpriteMeshType.FullRect, new Vector4(r + 1, r + 1, r + 1, r + 1));
        }

        public static Sprite Round => round ? round : (round = MakeRound(true));
        public static Sprite Ring => ring ? ring : (ring = MakeRound(false));
        public static Sprite Dot
        {
            get
            {
                if (dot) return dot;
                const int n = 32; var t = new Texture2D(n, n, TextureFormat.RGBA32, false);
                for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
                    { float d = Vector2.Distance(new Vector2(x + .5f, y + .5f), new Vector2(n / 2f, n / 2f)); t.SetPixel(x, y, new Color(1, 1, 1, Mathf.Clamp01(n / 2f - d))); }
                t.Apply();
                return dot = Sprite.Create(t, new Rect(0, 0, n, n), new Vector2(.5f, .5f));
            }
        }

        public static Image Fill(RectTransform r, Color c, bool rounded = true)
        {
            var i = r.gameObject.AddComponent<Image>(); i.color = c;
            if (rounded) { i.sprite = Round; i.type = Image.Type.Sliced; }
            return i;
        }

        // 테두리 (안쪽이 비어 있어 뒤가 보임)
        public static Image Border(RectTransform r, Color c)
        {
            var b = Ui.NewRect("border", r); Ui.Anchor(b, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var i = b.gameObject.AddComponent<Image>(); i.sprite = Ring; i.type = Image.Type.Sliced; i.color = c; i.raycastTarget = false;
            b.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
            return i;
        }

        public static Text T(Transform p, string s, int size, Color? c = null, bool bold = false, TextAnchor a = TextAnchor.MiddleLeft, float h = -1)
        {
            var t = Ui.Label(p, s, size, c ?? INK, bold ? FontStyle.Bold : FontStyle.Normal, a, h);
            t.raycastTarget = false;
            return t;
        }

        public static RectTransform Line(Transform p, float h = 1, Color? c = null)
        {
            var r = Ui.NewRect("line", p); Fill(r, c ?? LINE, false); Ui.Pref(r, h: h); return r;
        }

        public static RectTransform Space(Transform p, float h, bool flex = false)
        {
            var r = Ui.NewRect("space", p); var le = Ui.Pref(r, h: h); if (flex) le.flexibleHeight = 1; return r;
        }

        // 패널: 흰 바탕 + 선, 위에 연회색 제목 줄 (웹 .panel / .panel-heading)
        public static RectTransform Panel(Transform p, string name, bool fill = true)
        {
            var r = Ui.VBox(p, name, 0, 0);
            if (fill) Fill(r, Color.white); else { var i = r.gameObject.AddComponent<Image>(); i.color = new Color(0, 0, 0, 0); i.raycastTarget = false; }
            Border(r, LINE);
            return r;
        }

        public static (RectTransform bar, Text left, Text right) Heading(Transform panel, string left, string right = null, bool tiny = false, float h = 56, int pad = 24)
        {
            var bar = Ui.HBox(panel, "heading", 8, h);
            var hg = bar.GetComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(pad, pad, 0, 0); hg.childForceExpandWidth = false; hg.childForceExpandHeight = false;
            Fill(bar, HEAD, false);
            var l = T(bar, left, 14, ACCENT, true); Ui.Pref(l.rectTransform, flexW: 1);
            Text rt = null;
            if (right != null)
            {
                if (tiny)
                {
                    var box = Ui.NewRect("tiny", bar); var hb = box.gameObject.AddComponent<HorizontalLayoutGroup>(); hb.padding = new RectOffset(9, 9, 4, 4);
                    hb.childControlWidth = hb.childControlHeight = true; hb.childForceExpandWidth = false;
                    var bi = Fill(box, Color.white); Border(box, LINE); bi.color = new Color(1, 1, 1, 0);
                    rt = T(box, right, 12, MUTED); Ui.Pref(box, h: 26).flexibleHeight = 0;
                }
                else rt = T(bar, right, 12, MUTED, false, TextAnchor.MiddleRight);
            }
            Line(panel);
            return (bar, l, rt);
        }

        public enum B { Primary, Secondary, Text, Stop, Danger, Console }

        public static Button Btn(Transform p, string s, Action click, B kind = B.Secondary, float h = 48, int size = 15, float w = -1)
        {
            var r = Ui.NewRect("btn_" + s, p);
            Image img;
            if (kind == B.Text || kind == B.Stop || kind == B.Danger) { img = r.gameObject.AddComponent<Image>(); img.color = new Color(1, 1, 1, 0); }
            else
            {
                img = Fill(r, kind == B.Primary ? ACCENT : kind == B.Console ? Color.white : SURFACE2);
                if (kind != B.Primary) Border(r, kind == B.Console ? Hex(0xe3b3b8) : BTN_LINE);
            }
            var b = r.gameObject.AddComponent<Button>(); b.targetGraphic = img;
            var cb = b.colors; cb.highlightedColor = new Color(1.06f, 1.06f, 1.06f); cb.pressedColor = new Color(.88f, .9f, .95f); cb.disabledColor = Color.white; cb.fadeDuration = .08f; b.colors = cb;
            Color fg = kind == B.Primary ? Color.white : kind == B.Text ? MUTED : kind == B.Stop || kind == B.Console ? STOP : kind == B.Danger ? RED : INK;
            var t = T(r, s, size, fg, kind != B.Text, kind == B.Text ? TextAnchor.MiddleLeft : TextAnchor.MiddleCenter);
            Ui.Anchor(t.rectTransform, Vector2.zero, Vector2.one, new Vector2(6, 0), new Vector2(-6, 0));
            b.onClick.AddListener(() => click());
            var le = Ui.Pref(r, w, h);
            if (w >= 0) { le.minWidth = w; le.flexibleWidth = 0; }
            // 비활성 때 글자도 흐리게 (웹 opacity .35)
            r.gameObject.AddComponent<FadeWhenDisabled>();
            return b;
        }

        // 이름이 위에 있는 입력 칸 (웹 label > input)
        public static InputField Field(Transform p, string label, string value, float w = -1, float h = 44)
        {
            var box = Ui.VBox(p, "field_" + label, 0, 7);
            if (w >= 0) { var le = Ui.Pref(box, w); le.minWidth = w; le.flexibleWidth = 0; } else Ui.Pref(box, 0, -1, 1);
            T(box, label, 13, MUTED, false, TextAnchor.LowerLeft, 18);
            return Input(box, value, h);
        }

        public static InputField Input(Transform p, string value, float h = 44, TextAnchor align = TextAnchor.MiddleLeft)
        {
            var r = Ui.NewRect("input", p);
            Fill(r, Color.white); Border(r, INPUT_LINE);
            var text = Ui.Label(r, "", 15, INK, FontStyle.Normal, align);
            text.supportRichText = false; text.horizontalOverflow = HorizontalWrapMode.Overflow;
            Ui.Anchor(text.rectTransform, Vector2.zero, Vector2.one, new Vector2(13, 2), new Vector2(-13, -2));
            var f = r.gameObject.AddComponent<InputField>();
            f.textComponent = text; f.text = value; f.lineType = InputField.LineType.SingleLine; f.caretColor = INK; f.selectionColor = new Color(.04f, .24f, .57f, .25f);
            Ui.Pref(r, 0, h, 1);
            return f;
        }

        // 선택 상자 대신 단추 묶음 (웹 select)
        public static List<Button> Segmented(Transform p, string[] options, int selected, Action<int> changed, float h = 44, int size = 13)
        {
            var row = Ui.HBox(p, "segmented", 0, h);
            var list = new List<Button>();
            for (int i = 0; i < options.Length; i++)
            {
                int k = i;
                var b = Btn(row, options[i], () => { SetSegment(list, k); changed(k); }, B.Secondary, h, size);
                list.Add(b);
            }
            SetSegment(list, selected);
            return list;
        }

        public static void SetSegment(List<Button> list, int k)
        {
            for (int i = 0; i < list.Count; i++)
            {
                var img = list[i].GetComponent<Image>(); img.color = i == k ? PILL_BG : Color.white;
                var t = list[i].GetComponentInChildren<Text>(); t.color = i == k ? ACCENT : MUTED;
                var bd = list[i].transform.Find("border")?.GetComponent<Image>(); if (bd) bd.color = i == k ? ACCENT : INPUT_LINE;
            }
        }

        // 모드 표시 (웹 .pill)
        public static Text Pill(Transform p)
        {
            var r = Ui.NewRect("pill", p);
            var hg = r.gameObject.AddComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(16, 18, 10, 10); hg.spacing = 10; hg.childAlignment = TextAnchor.MiddleLeft;
            hg.childControlWidth = hg.childControlHeight = true; hg.childForceExpandWidth = hg.childForceExpandHeight = false;
            Fill(r, PILL_BG); Border(r, PILL_LINE);
            var d = Ui.NewRect("dot", r); var di = d.gameObject.AddComponent<Image>(); di.sprite = Dot; di.color = ACCENT; Ui.Pref(d, 7, 7);
            var t = T(r, "연결 확인 중", 14, ACCENT);
            Ui.Pref(r, h: 40);
            return t;
        }

        public static Toggle Check(Transform p, string text, Action<bool> changed, bool on = false)
        {
            var r = Ui.HBox(p, "check", 9, 26);
            var hg = r.GetComponent<HorizontalLayoutGroup>(); hg.childForceExpandWidth = false; hg.childForceExpandHeight = false;
            var box = Ui.NewRect("box", r); Fill(box, Color.white); Border(box, INPUT_LINE); Ui.Pref(box, 20, 20);
            var mark = Ui.NewRect("mark", box); var mi = Fill(mark, ACCENT); Ui.Anchor(mark, Vector2.zero, Vector2.one, new Vector2(4, 4), new Vector2(-4, -4));
            var t = T(r, text, 13, INK); Ui.Pref(t.rectTransform, flexW: 1);
            var tg = r.gameObject.AddComponent<Toggle>(); tg.targetGraphic = box.GetComponent<Image>(); tg.graphic = mi; tg.isOn = on;
            tg.onValueChanged.AddListener(v => changed(v));
            return tg;
        }

        public static RectTransform Row(Transform p, float spacing = 10, float h = 48, bool expand = true)
        {
            var r = Ui.HBox(p, "row", spacing, h);
            var g = r.GetComponent<HorizontalLayoutGroup>(); g.childForceExpandWidth = expand; g.childForceExpandHeight = false;
            return r;
        }

        // 화면 아래 알림 (웹 #toast)
        static RectTransform toastRoot; static Text toastText; static Image toastBg; static float toastUntil;
        public static void InitToast(RectTransform canvasRoot)
        {
            toastRoot = Ui.NewRect("toast", canvasRoot);
            toastRoot.anchorMin = toastRoot.anchorMax = new Vector2(.5f, 0); toastRoot.pivot = new Vector2(.5f, 0); toastRoot.anchoredPosition = new Vector2(0, 30);
            var hg = toastRoot.gameObject.AddComponent<HorizontalLayoutGroup>(); hg.padding = new RectOffset(26, 26, 16, 16); hg.childControlWidth = hg.childControlHeight = true;
            var f = toastRoot.gameObject.AddComponent<ContentSizeFitter>(); f.horizontalFit = f.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            toastBg = Fill(toastRoot, ACCENT); toastBg.raycastTarget = false;
            toastText = T(toastRoot, "", 15, Color.white); toastText.horizontalOverflow = HorizontalWrapMode.Overflow;
            toastRoot.gameObject.SetActive(false);
        }

        public static void Toast(string msg, bool error = false)
        {
            if (toastRoot == null) return;
            toastText.text = msg; toastText.color = error ? Hex(0x9d2633) : Color.white;
            toastBg.color = error ? Hex(0xfff0f1) : ACCENT;
            toastRoot.SetAsLastSibling(); toastRoot.gameObject.SetActive(true);
            toastUntil = Time.unscaledTime + (error ? 8 : 4);
        }

        public static void TickToast() { if (toastRoot != null && toastRoot.gameObject.activeSelf && Time.unscaledTime > toastUntil) toastRoot.gameObject.SetActive(false); }

        // 실행 중 오류를 알림으로 (웹 action())
        public static void Act(Action f)
        {
            try { f(); }
            catch (Problem e) { Toast(e.Message, true); }
            catch (RunError e) { Toast(e.Message, true); }
            catch (Exception e) { Debug.LogException(e); Toast("요청을 처리하지 못했습니다.", true); }
        }

        // 확인 창 (웹 confirm())
        static RectTransform modal; static Text modalText; static Action modalOk;
        public static void InitModal(RectTransform canvasRoot)
        {
            modal = Ui.NewRect("modal", canvasRoot); Ui.Anchor(modal, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var dim = modal.gameObject.AddComponent<Image>(); dim.color = new Color(.1f, .14f, .2f, .35f);
            var card = Ui.VBox(modal, "card", 28, 18); card.anchorMin = card.anchorMax = new Vector2(.5f, .5f); card.pivot = new Vector2(.5f, .5f); card.sizeDelta = new Vector2(560, 0);
            card.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            Fill(card, Color.white); Border(card, LINE);
            modalText = T(card, "", 16, INK); modalText.horizontalOverflow = HorizontalWrapMode.Wrap;
            var row = Row(card, 10, 46);
            Btn(row, "취소", () => modal.gameObject.SetActive(false), B.Secondary, 46);
            Btn(row, "확인", () => { modal.gameObject.SetActive(false); modalOk?.Invoke(); }, B.Primary, 46);
            modal.gameObject.SetActive(false);
        }

        public static void Confirm(string msg, Action ok)
        {
            modalText.text = msg; modalOk = ok; modal.SetAsLastSibling(); modal.gameObject.SetActive(true);
        }

        public static string Mm(int v) => (v / 1000.0).ToString("0.000");

        // 빨간 십자 (웹 .crosshair)
        public static void Crosshair(RectTransform view, float len)
        {
            foreach (var (w, h) in new[] { (len, 2f), (2f, len) })
            {
                var r = Ui.NewRect("cross", view); r.anchorMin = r.anchorMax = new Vector2(.5f, .5f); r.sizeDelta = new Vector2(w, h);
                Fill(r, RED, false).raycastTarget = false;
            }
        }
    }

    public class Problem : Exception { public Problem(string m) : base(m) { } }

    // 버튼 비활성 때 안쪽 글자까지 흐리게
    public class FadeWhenDisabled : MonoBehaviour
    {
        Selectable s; CanvasGroup g; bool last = true;
        void Awake() { s = GetComponent<Selectable>(); g = gameObject.AddComponent<CanvasGroup>(); }
        void LateUpdate() { if (s == null) return; bool on = s.IsInteractable(); if (on != last) { g.alpha = on ? 1 : .35f; last = on; } }
    }
}
