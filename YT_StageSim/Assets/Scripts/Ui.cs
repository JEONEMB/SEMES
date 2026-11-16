// uGUI 를 코드로 만드는 도우미 (장면 파일 없이 실행 시 생성)
using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace YT
{
    public static class Ui
    {
        public static Font font;
        public static readonly Color Ink = new Color(0.13f, 0.15f, 0.17f), Muted = new Color(0.40f, 0.43f, 0.46f),
            Panel = new Color(0.975f, 0.975f, 0.965f, 0.97f), Card = new Color(1f, 1f, 1f, 1f), Line = new Color(0.86f, 0.87f, 0.85f),
            Accent = new Color(0.11f, 0.37f, 0.82f), AccentSoft = new Color(0.88f, 0.92f, 0.99f), Danger = new Color(0.78f, 0.22f, 0.17f),
            Good = new Color(0.12f, 0.48f, 0.30f), Btn = new Color(0.93f, 0.94f, 0.95f);

        public static Font GetFont()
        {
            if (font == null) font = UnityEngine.Font.CreateDynamicFontFromOSFont(new[] { "Malgun Gothic", "맑은 고딕", "Segoe UI", "Arial" }, 16);
            return font;
        }

        public static Canvas NewCanvas(string name)
        {
            var go = new GameObject(name, typeof(RectTransform));
            var c = go.AddComponent<Canvas>(); c.renderMode = RenderMode.ScreenSpaceOverlay;
            var s = go.AddComponent<CanvasScaler>(); s.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize; s.referenceResolution = new Vector2(1920, 1080); s.matchWidthOrHeight = 0.5f;
            go.AddComponent<GraphicRaycaster>();
            if (UnityEngine.Object.FindAnyObjectByType<EventSystem>() == null)
                new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
            return c;
        }

        public static RectTransform NewRect(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return (RectTransform)go.transform;
        }

        public static RectTransform Anchor(RectTransform r, Vector2 min, Vector2 max, Vector2 offMin, Vector2 offMax)
        {
            r.anchorMin = min; r.anchorMax = max; r.offsetMin = offMin; r.offsetMax = offMax; return r;
        }

        public static Image Bg(RectTransform r, Color c) { var i = r.gameObject.AddComponent<Image>(); i.color = c; return i; }

        public static Text Label(Transform parent, string text, int size = 15, Color? color = null, FontStyle style = FontStyle.Normal, TextAnchor anchor = TextAnchor.MiddleLeft, float h = -1)
        {
            var r = NewRect("label", parent);
            var t = r.gameObject.AddComponent<Text>();
            t.font = GetFont(); t.fontSize = size; t.color = color ?? Ink; t.fontStyle = style; t.alignment = anchor; t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.verticalOverflow = VerticalWrapMode.Overflow; t.supportRichText = true;
            if (h > 0) Pref(r, h: h);
            return t;
        }

        public static LayoutElement Pref(RectTransform r, float w = -1, float h = -1, float flexW = -1)
        {
            var le = r.GetComponent<LayoutElement>();
            if (le == null) le = r.gameObject.AddComponent<LayoutElement>();
            if (w >= 0) le.preferredWidth = w;
            if (h >= 0) { le.preferredHeight = h; le.minHeight = h; }
            if (flexW >= 0) le.flexibleWidth = flexW;
            return le;
        }

        public static RectTransform VBox(Transform parent, string name, int pad = 0, float spacing = 6)
        {
            var r = NewRect(name, parent);
            var v = r.gameObject.AddComponent<VerticalLayoutGroup>();
            v.padding = new RectOffset(pad, pad, pad, pad); v.spacing = spacing;
            v.childControlWidth = v.childControlHeight = true; v.childForceExpandWidth = true; v.childForceExpandHeight = false;
            return r;
        }

        public static RectTransform HBox(Transform parent, string name, float spacing = 6, float h = 34)
        {
            var r = NewRect(name, parent);
            var g = r.gameObject.AddComponent<HorizontalLayoutGroup>();
            g.spacing = spacing; g.childControlWidth = g.childControlHeight = true; g.childForceExpandWidth = true; g.childForceExpandHeight = true;
            g.childAlignment = TextAnchor.MiddleLeft;
            Pref(r, h: h).flexibleHeight = 0;   // 가로 줄은 세로로 늘어나지 않게 (childForceExpandHeight 가 부모에 flexible 1 을 전달하는 것 막음)
            return r;
        }

        // 제목이 있는 카드
        public static RectTransform Section(Transform parent, string title)
        {
            var card = VBox(parent, "section_" + title, 12, 6);
            Bg(card, Card);
            var o = card.gameObject.AddComponent<Outline>(); o.effectColor = Line; o.effectDistance = new Vector2(1, -1);
            Label(card, title, 16, Ink, FontStyle.Bold, TextAnchor.MiddleLeft, 24);
            return card;
        }

        public static Button NewButton(Transform parent, string text, Action onClick, Color? bg = null, Color? fg = null, float h = 34, int size = 15)
        {
            var r = NewRect("btn_" + text, parent);
            var img = Bg(r, bg ?? Btn);
            var b = r.gameObject.AddComponent<UnityEngine.UI.Button>();
            b.targetGraphic = img;
            var cb = b.colors; cb.highlightedColor = new Color(0.9f, 0.93f, 1f); cb.pressedColor = new Color(0.8f, 0.85f, 0.95f); cb.disabledColor = new Color(1, 1, 1, 0.45f); b.colors = cb;
            var t = Label(r, text, size, fg ?? Ink, FontStyle.Bold, TextAnchor.MiddleCenter);
            Anchor(t.rectTransform, Vector2.zero, Vector2.one, new Vector2(4, 0), new Vector2(-4, 0));
            b.onClick.AddListener(() => onClick());
            Pref(r, h: h);
            return b;
        }

        public static InputField NewInput(Transform parent, string value, float w = -1, float h = 32)
        {
            var r = NewRect("input", parent);
            Bg(r, Color.white);
            var o = r.gameObject.AddComponent<Outline>(); o.effectColor = new Color(0.72f, 0.74f, 0.76f); o.effectDistance = new Vector2(1, -1);
            var text = Label(r, "", 15, Ink, FontStyle.Normal, TextAnchor.MiddleRight);
            text.supportRichText = false; text.horizontalOverflow = HorizontalWrapMode.Overflow;
            Anchor(text.rectTransform, Vector2.zero, Vector2.one, new Vector2(8, 2), new Vector2(-8, -2));
            var f = r.gameObject.AddComponent<InputField>();
            f.textComponent = text; f.text = value; f.lineType = InputField.LineType.SingleLine;
            Pref(r, w, h, w < 0 ? 1 : 0);
            return f;
        }

        public static Toggle NewToggle(Transform parent, string text, bool on, Action<bool> changed)
        {
            var r = HBox(parent, "toggle_" + text, 8, 26);
            var hg = r.GetComponent<HorizontalLayoutGroup>(); hg.childForceExpandWidth = false;
            var box = NewRect("box", r); Bg(box, Color.white); Pref(box, 20, 20);
            var bo = box.gameObject.AddComponent<Outline>(); bo.effectColor = new Color(0.6f, 0.62f, 0.65f);
            var mark = NewRect("mark", box); var mi = Bg(mark, Accent); Anchor(mark, Vector2.zero, Vector2.one, new Vector2(4, 4), new Vector2(-4, -4));
            var t = Label(r, text, 14); Pref(t.rectTransform, flexW: 1);
            var tg = r.gameObject.AddComponent<UnityEngine.UI.Toggle>();
            tg.targetGraphic = box.GetComponent<Image>(); tg.graphic = mi; tg.isOn = on;
            tg.onValueChanged.AddListener(v => changed(v));
            return tg;
        }

        public static Slider NewSlider(Transform parent, float min, float max, float value, Action<float> changed, float h = 22)
        {
            var r = NewRect("slider", parent); Pref(r, h: h, flexW: 1);
            var bg = NewRect("bg", r); Anchor(bg, new Vector2(0, 0.5f), new Vector2(1, 0.5f), new Vector2(0, -3), new Vector2(0, 3)); Bg(bg, new Color(0.82f, 0.84f, 0.86f));
            var fa = NewRect("fill_area", r); Anchor(fa, new Vector2(0, 0.5f), new Vector2(1, 0.5f), new Vector2(0, -3), new Vector2(0, 3));
            var fill = NewRect("fill", fa); Bg(fill, Accent); Anchor(fill, Vector2.zero, new Vector2(0, 1), Vector2.zero, Vector2.zero);
            var ha = NewRect("handle_area", r); Anchor(ha, Vector2.zero, Vector2.one, new Vector2(8, 0), new Vector2(-8, 0));
            var hd = NewRect("handle", ha); hd.sizeDelta = new Vector2(16, 0); var hi = Bg(hd, Color.white);
            var ho = hd.gameObject.AddComponent<Outline>(); ho.effectColor = Accent;
            var s = r.gameObject.AddComponent<Slider>();
            s.fillRect = fill; s.handleRect = hd; s.targetGraphic = hi; s.direction = Slider.Direction.LeftToRight;
            s.minValue = min; s.maxValue = max; s.value = value;
            s.onValueChanged.AddListener(v => changed(v));
            return s;
        }

        // 3 자리 소수 (웹 units() 와 같은 규칙: 소수점 셋째 자리까지만)
        public static bool ParseMilli(string s, out int milli, out string err)
        {
            milli = 0; err = null;
            if (!double.TryParse(s.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) || double.IsNaN(v) || double.IsInfinity(v))
            { err = "숫자를 입력하세요."; return false; }
            double sc = v * 1000;
            if (Math.Abs(sc - Math.Round(sc)) > 1e-6) { err = "수치는 소수점 셋째 자리까지 입력하세요."; return false; }
            milli = (int)Math.Round(sc); return true;
        }
    }
    // 끌어서 옮기는 창: 제목 줄에 붙이고 target 을 옮김. 부모 영역(프로그램 창) 밖으로는 나가지 않게 막음
    public class DragWindow : MonoBehaviour, IBeginDragHandler, IDragHandler
    {
        public RectTransform target;
        Canvas canvas;
        static readonly Vector3[] C = new Vector3[4], Pc = new Vector3[4];

        public void OnBeginDrag(PointerEventData e)
        {
            canvas = target.GetComponentInParent<Canvas>();
            target.SetAsLastSibling();
        }

        public void OnDrag(PointerEventData e)
        {
            float k = canvas != null ? canvas.scaleFactor : 1f;
            target.anchoredPosition += e.delta / k;
            Clamp();
        }

        public void Clamp()
        {
            var parent = target.parent as RectTransform;
            if (parent == null) return;
            target.GetWorldCorners(C); parent.GetWorldCorners(Pc);
            var d = Vector3.zero;
            if (C[0].x < Pc[0].x) d.x = Pc[0].x - C[0].x; else if (C[2].x > Pc[2].x) d.x = Pc[2].x - C[2].x;
            if (C[0].y < Pc[0].y) d.y = Pc[0].y - C[0].y; else if (C[2].y > Pc[2].y) d.y = Pc[2].y - C[2].y;
            target.position += d;
        }
    }
}
