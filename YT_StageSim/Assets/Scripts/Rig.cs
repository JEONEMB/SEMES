// 장비 3D 모델 (도면 = zip index.html, 출력 부품·타깃 = GitLab dev hardware/3d-print STL)
// 계층: Rig(고정) ─ YCarriage(Y 로 움직임) ─ Theta(Θ 로 돎, 회전축 = 원점 수직선)
// 도면 기준 자세: Y = 0 이 스테이지 가운데, Θ = 0 이 카메라가 QR 을 정면으로 보는 자세 (펌웨어 좌표와 다름: Stage.cs 참고)
using System;
using System.Collections.Generic;
using UnityEngine;
using static YT.Geo;

namespace YT
{
    public class PartInfo
    {
        public string id, no, name, kind, size, where, how;
        public readonly List<Renderer> renderers = new List<Renderer>();
    }

    public class Rig
    {
        public const float QRX = 75f + 307.8f;     // QR 플레이트 앞면 (렌즈 앞면 75 + WD 307.8)
        public const float BGX = QRX + 88f;        // 배경판 앞면
        public const float CY = 251f;              // 렌즈 중심 높이 (3D 도면값. GitLab qr_target README: 조립 후 실측 전)
        public const float TARGET_LIFT = 25.5f;    // 타깃 전체를 올린 높이 (GitLab dev cda0d03, 2026-11-12, make_target_board.py LIFT)
        public const float QR_H = CY + TARGET_LIFT; // QR 플레이트 중심 높이 (타깃 좌표 LENS_H 251 + LIFT)
        public const float Y_SWITCH_ON = 22.4f;    // Y 끝 스위치가 켜지는 위치 ±22.4 (도면 설명)
        public const float Y_STROKE = 25f;         // 스테이지 행정 ±25
        public const float TH_SWITCH_ON = -1.7f;   // Θ 원점 스위치가 켜지는 각 (도면 설명, −1.7° 에서 +15° 까지 눌림)

        public Transform root, yG, tG, camMount;
        public readonly Dictionary<string, PartInfo> parts = new Dictionary<string, PartInfo>();
        readonly Dictionary<Renderer, string> owner = new Dictionary<Renderer, string>();
        public LineRenderer fovLines;

        // 색 (도면과 같음)
        const uint PLY = 0xd9b882, LAUAN = 0xc48a55, PLA = 0xf0ac1c, PLA2 = 0xe39a12, ALU = 0xb9c0c7, BLACK = 0x2a2d31, STEEL = 0x8c939a,
            STAGE = 0x3a3f46, WHITE = 0xf2efe6, DESK = 0xd8d2c6, WALL = 0xf1f1ec, RED = 0xc0392b, GREEN = 0x1f7a4d, FR = 0xc39a55, TCOL = 0xefece4;

        static readonly string[][] CAT =
        {
            new[] {"base", "1", "바닥판", "wood", "자작합판 9T 300×400", "책상 앞 끝에 맞춤", "양 끝을 C클램프로 책상과 함께 물림. Y 스테이지는 바닥판 아래에서 M3 볼트로 고정"},
            new[] {"cclamp", "2", "C클램프 ×2", "buy", "SK11 C클램프 100mm", "바닥판 사람 쪽 양 끝 (스테이지 중심에서 130)", "바닥판과 책상 상판을 함께 조임"},
            new[] {"stage", "3", "Y 스테이지 LSM-NK223", "buy", "T6×2-50mm · 폭 30 · 길이 126", "바닥판 위, 벽과 나란히", "밑면 M3 탭 ×4에 바닥판 아래에서 볼트 (형상은 개략)"},
            new[] {"carriage", "4", "캐리지 (움직이는 부분)", "buy", "32×30, M3 탭 ×4 (20×20)", "스테이지 위", "행정 ±25mm"},
            new[] {"spacer", "5", "PA 스페이서 10 + M3×22 ×4", "buy", "내경3×외경6×10", "캐리지와 플랫폼판 사이", "판을 끝판보다 3.5mm 띄움"},
            new[] {"platform", "6", "플랫폼판", "wood", "자작합판 9T 150×120", "캐리지 위 (150이 Y 방향)", "가운데 20×20 Ø4 ×4"},
            new[] {"pbase", "7", "모터 받침 (U자 클램프·원점 기둥 일체)", "print", "STL 07 v5 (GitLab)", "플랫폼판 위 가운데", "관통 M3×20 ×4 + 판 아래 너트"},
            new[] {"pclamp", "8", "클램프 덮개 (벽 쪽)", "print", "STL 08 (GitLab)", "모터 벽 쪽", "M3 ×4로 ⑦ 팔 끝 귀에 조임"},
            new[] {"motor", "9", "하모닉 모터 PSH42 1/100", "buy", "STL 09 모형 v2 (GitLab)", "받침 위에 세움, 출력축 위", "U자 클램프와 덮개로 고정"},
            new[] {"coupling", "10", "플랜지 커플링 내경 10", "print", "STL 24 (GitLab, 키홈)", "출력축 위", "축에 무두볼트, 플랜지는 각재 아랫면에 태핑나사"},
            new[] {"beam", "11", "카메라 각재", "wood", "라왕 30×30×120", "커플링 위, Y 방향으로 가로", "Θ로 함께 도는 팔"},
            new[] {"cplate", "12", "카메라판", "wood", "자작합판 9T 100×100", "각재의 벽 쪽 면에 세움", "목공 나사 ×2로 각재에 고정"},
            new[] {"camera", "13", "HQ 카메라 + 6mm CS 렌즈", "buy", "보드 38×38 · 렌즈 앞면 축에서 약 75 (형상은 개략)", "카메라판 벽 쪽, 렌즈가 벽을 봄", "M2.5×20 ×4"},
            new[] {"homesw", "14", "Θ 원점 스위치 KW-10 + 감지돌기", "print", "감지돌기 STL 14 v5 (GitLab), 스위치는 데이터시트 치수", "⑦ 기둥 판 바깥(−Y) 면", "−1.7°에서 켜짐 (도면 설명). 펌웨어 0.3.1 은 Θ 수동 원점"},
            new[] {"ysw", "15", "Y 끝 스위치 KW-10 ×2 + 감지돌기", "print", "브래킷 STL 15 v5 ×2, 감지돌기 STL 15 v4 (GitLab)", "X −53, Y ±15..±35", "Y ≈ ±22.4에서 켬. −Y 가 원점, +Y 는 끝 스위치"},
            new[] {"elec", "16", "제어 기판·전원", "buy", "NUCLEO-F446RE, 만능기판 ×2, LM2596, WAGO (개략)", "바닥판 사람 쪽", "M3 + PA 스페이서"},
            new[] {"qr", "17", "QR 플레이트 69×15", "print", "STL 17 (GitLab) + QR 라벨", "렌즈 앞면에서 307.8 (WD)", "M3×12 ×2로 ⑳ 받침에 조임"},
            new[] {"pihold", "18", "라즈베리파이 5 + 홀더", "print", "홀더 STL 18 (GitLab), Pi 는 상자로 단순화", "플랫폼판 −Y 끝 (Y와 같이 움직임)", "M3×20 ×2 + 판 아래 너트"},
            new[] {"tpanel", "19", "배경판 (포맥스 5T) + 배경 나사", "buy", "STL 19 (GitLab) 400×360×5", "플레이트 뒤 88", "배경 나사 M3 ×4 (김성현 W2·W3 자리 포함)"},
            new[] {"tbracket", "20", "플레이트 받침", "print", "STL 20 (GitLab)", "배경판 가운데, 플레이트를 88 앞으로 내밈", "날개를 판에 M4 ×4"},
            new[] {"tstand", "21", "옆 받침 ×2", "print", "STL 21 왼쪽·오른쪽 (GitLab, 2026-11-12 판 전체 25.5 mm 올림)", "판 양 끝", "판에 M4×20 ×2씩, 발 뒤 끝을 책상 모서리에"},
            new[] {"tclamp", "22", "C클램프 ×2 (타깃)", "buy", "SK11 C클램프 100mm", "옆 받침 발 뒤 끝", "옆 받침 발과 책상 상판을 함께 조임"},
            new[] {"tstrip", "23", "위 턱 띠 · 아래 단 띠", "print", "STL 23 위·아래 (GitLab)", "판 앞면", "양면테이프 또는 순간접착제"},
            new[] {"room", "", "책상·벽 (참고)", "ref", "책상 깊이 ≈ 833 (이 배치 예시)", "", ""},
        };

        public static readonly Dictionary<string, string> KIND = new Dictionary<string, string>
            { {"buy", "신청서 구매"}, {"wood", "합판·각재 주문"}, {"print", "팀 3D 프린터 출력"}, {"ref", "참고"} };

        // 분해 보기: 도면 G(id, parent, {x, y}) 의 ex 값 (분해 100% 일 때 그 묶음이 움직이는 거리, 도면 x·높이 방향)
        public readonly List<(Transform g, float ex, float ey)> explode = new List<(Transform, float, float)>();
        public readonly Dictionary<string, Transform> anchors = new Dictionary<string, Transform>();

        Transform G(string id, Transform parent, float ex = 0, float ey = 0)
        {
            var g = Group(id, parent);
            g.gameObject.AddComponent<PartTag>().id = id;
            if (ex != 0 || ey != 0) explode.Add((g, ex, ey));
            return g;
        }

        // 부품 번호 풍선 자리 (도면 anchor)
        void A(string id, Transform g, float x, float y, float z)
        {
            var a = Group("anchor_" + id, g);
            a.localPosition = P(x, y, z);
            if (!anchors.ContainsKey(id)) anchors[id] = a;
        }

        public void SetExplode(float k)
        {
            foreach (var e in explode) e.g.localPosition = new Vector3(e.ex * k, e.ey * k, 0);
        }

        public string PartOf(Renderer r) => r != null && owner.TryGetValue(r, out var id) ? id : null;

        public static Rig Build(Transform parent)
        {
            var r = new Rig();
            r.Make(parent);
            return r;
        }

        void Make(Transform parent)
        {
            foreach (var c in CAT) parts[c[0]] = new PartInfo { id = c[0], no = c[1], name = c[2], kind = c[3], size = c[4], where = c[5], how = c[6] };
            root = Group("Rig", parent);
            yG = Group("YCarriage", root);
            tG = Group("Theta", yG);
            var metal = Metal(STEEL);

            // 책상·벽
            const float DESK_END = 616, WALLX = 680;
            var gRoom = G("room", root);
            Box(gRoom, Mat(DESK), -217, DESK_END, -25, 0, -550, 550, "desk");
            var wall = GameObject.CreatePrimitive(PrimitiveType.Quad);
            wall.name = "wall"; wall.transform.SetParent(gRoom, false);
            wall.transform.localPosition = P(WALLX, 355, 0);
            wall.transform.localRotation = Quaternion.AngleAxis(90, Vector3.up);   // 면이 카메라 쪽(-x)
            wall.transform.localScale = new Vector3(1100, 760, 1);
            wall.GetComponent<Renderer>().sharedMaterial = Mat(WALL, 0.95f);

            // 타깃 (STL 좌표: X = 렌즈 앞면에서 벽 쪽, Y = 카메라에서 볼 때 왼쪽, Z = 책상 위 → 도면 (75 + X, Z, −Y))
            Func<Vector3, Vector3> tmap = s => new Vector3(75 + s.x, s.z + TARGET_LIFT, -s.y);   // 옆 받침 STL 은 책상 면이 −LIFT → 전체 +LIFT
            var tmat = Mat(TCOL, 0.88f);
            var gTp = G("tpanel", root, 40); A("tpanel", gTp, BGX, 345, -150);
            Stl.Part(gTp, "19_background_board_400x360x5.stl", Mat(0xf7f7f4, 0.88f), tmap);
            foreach (var yz in new[] { new Vector2(85.8f, 228.2f), new Vector2(-84.7f, 227.7f), new Vector2(81.3f, 200.9f), new Vector2(-80.5f, 200.3f) })
                CylX(gTp, metal, 2.85f, yz.y + TARGET_LIFT, -yz.x, BGX - 1.65f, BGX, "screw_M3");
            var gTb = G("tbracket", root, -60); A("tbracket", gTb, QRX + 45, 268, 35);
            Stl.Part(gTb, "20_plate_bracket.stl", tmat, tmap);
            var gTr = G("tstrip", root, -25); A("tstrip", gTr, BGX - 6, 290, 120);
            Stl.Part(gTr, "23_strip_upper_400.stl", tmat, tmap);
            Stl.Part(gTr, "23_strip_lower_400.stl", tmat, tmap);
            var gTs = G("tstand", root, 110); A("tstand", gTs, BGX + 20, 230, -205);
            Stl.Part(gTs, "21_side_stand_left.stl", tmat, tmap);
            Stl.Part(gTs, "21_side_stand_right.stl", tmat, tmap);
            var gq = G("qr", root, -130); A("qr", gq, QRX - 1, 262, 26);
            Stl.Part(gq, "17_qr_plate_69x15.stl", tmat, tmap);
            {
                var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
                q.name = "qr_label"; q.transform.SetParent(gq, false);
                q.transform.localPosition = P(QRX - 0.05f, QR_H, 0);
                q.transform.localRotation = Quaternion.AngleAxis(90, Vector3.up);
                q.transform.localScale = new Vector3(69, 15, 1);
                var tex = new Texture2D(2, 2);
                tex.LoadImage(System.IO.File.ReadAllBytes(System.IO.Path.Combine(Application.streamingAssetsPath, "qr_plate.png")));
                tex.anisoLevel = 8;
                var qm = new Material(Shader.Find("Unlit/Transparent")) { mainTexture = tex, name = "qr_label" };
                q.GetComponent<Renderer>().sharedMaterial = qm;
                Kill(q.GetComponent<Collider>());
                foreach (var z in new[] { -29.3f, 29.3f }) CylX(gq, metal, 2.85f, QR_H, z, QRX - 1.65f, QRX - 0.1f, "screw_M3");
            }
            var gTc = G("tclamp", root, 150); A("tclamp", gTc, 610, 26, -194);
            foreach (var z in new[] { -194f, 194f })   // 책상 뒤 모서리(616)에서 옆 받침 발을 묾
                CClamp(gTc, z, new[] { 628f, 640, -60, 22 }, new[] { 570f, 640, 10, 22 }, new[] { 570f, 640, -60, -48 }, 591);

            // 1 바닥판
            var gBase = G("base", root); A("base", gBase, 150, 9, -120);
            Box(gBase, Mat(PLY), -217, 183, 0, 9, -150, 150, "plywood");
            // 2 C클램프
            var gCl = G("cclamp", root); A("cclamp", gCl, -228, 21, -130);
            foreach (var z in new[] { -130f, 130f })
                CClamp(gCl, z, new[] { -262f, -250, -60, 21 }, new[] { -262f, -197, 9, 21 }, new[] { -262f, -195, -60, -48 }, -203);

            // 3 Y 스테이지 본체 (고정, 개략)
            var gSt = G("stage", root); A("stage", gSt, 15, 40, 63);
            var st = Mat(STAGE);
            Box(gSt, st, -15, 15, 9, 15, -41, 41);
            Box(gSt, st, -15, -12, 15, 27, -41, 41);
            Box(gSt, st, 12, 15, 15, 27, -41, 41);
            CylZ(gSt, Metal(ALU), 3, 0, 21, -41, 41, "lead_screw");
            Box(gSt, st, -15, 15, 9, 37.5f, -49, -41);
            Box(gSt, st, -15, 15, 9, 44, 41, 49);
            Box(gSt, Mat(BLACK), -10, 10, 23.5f, 43.5f, 49, 77, "stage_motor");

            // 4 캐리지
            var gCa = G("carriage", yG, 0, 20); A("carriage", gCa, 15, 32, -16);
            Box(gCa, Metal(ALU), -15, 15, 27, 37.5f, -16, 16);
            Box(gCa, Metal(ALU), -8, 8, 17, 27, -10, 10);
            // 5 스페이서 + M3×22
            var gSp = G("spacer", yG, 0, 45); A("spacer", gSp, 13, 42, -10);
            var gSpH = G("spacer", yG, 0, 100);   // 볼트 머리·와셔 (도면 gSpH)
            foreach (var xz in Quad(10, 10))
            {
                CylY(gSp, Mat(WHITE), 3, xz.x, xz.y, 37.5f, 47.5f, "spacer");
                CylY(gSpH, metal, 3.5f, xz.x, xz.y, 56.5f, 57, "washer");
                CylY(gSpH, Metal(0x3b3f44), 2.75f, xz.x, xz.y, 57, 60, "bolt_head");
            }
            // 6 플랫폼판 (구멍 생략)
            var gPl = G("platform", yG, 0, 70); A("platform", gPl, 60, 56.5f, -62);
            Box(gPl, Mat(PLY), -60, 60, 47.5f, 56.5f, -75, 75, "plywood");

            // 7 모터 받침 (STL 07: X → 도면 x, Y → 도면 z(+Y), Z → 판 위 높이)
            var gPb = G("pbase", yG, 0, 130); A("pbase", gPb, 45, 62.5f, -32);
            var gNut = G("pbase", yG, 0, 40);     // 판 아래 너트 (도면 gNut)
            Stl.Part(gPb, "07_motor_base_uclamp_v5.stl", Mat(PLA), s => new Vector3(s.x, 56.5f + s.z, s.y));
            // 관통 볼트: GitLab STL 07 v5 의 구멍 자리 (x ±40, z −17.5 / +32.5). 도면 코드의 ±25 와 다름 → GitLab 기준
            foreach (var xz in new[] { new Vector2(-40, -17.5f), new Vector2(40, -17.5f), new Vector2(-40, 32.5f), new Vector2(40, 32.5f) })
            {
                CylY(gPb, metal, 3.5f, xz.x, xz.y, 62.5f, 63, "washer");
                CylY(gPb, Metal(0x3b3f44), 2.75f, xz.x, xz.y, 63, 66, "bolt_head");
                CylY(gNut, metal, 3.2f, xz.x, xz.y, 45.1f, 47.5f, "nut");
            }
            // 8 클램프 덮개 (STL 08: 두께 Z → 도면 x 21.2..27.2, X → 높이 62.5..106.5, Y → 도면 z)
            var gPc = G("pclamp", yG, 45, 130); A("pclamp", gPc, 27, 96, 4);
            Stl.Part(gPc, "08_clamp_cover.stl", Mat(PLA2), s => new Vector3(21.2f + s.z, 84.5f + s.x, s.y));
            // 9 하모닉 모터 (STL 09 모형: 뒷면 Z 0 = 높이 62.5, 커넥터 +Y)
            var gMo = G("motor", yG, 0, 200); A("motor", gMo, -21, 135, -21);
            Stl.Part(gMo, "09_harmonic_motor_psh42_v2.stl", Mat(BLACK, 0.6f), s => new Vector3(s.x, 62.5f + s.z, s.y));

            // 10 커플링 (STL 24: 플랜지 윗면이 Z 0 → 높이 186 − Z)
            var gCp = G("coupling", tG, 0, 250); A("coupling", gCp, 16, 180, 6);
            Stl.Part(gCp, "24_flange_coupling_d10.stl", Mat(PLA), s => new Vector3(s.x, 186 - s.z, -s.y));
            // 11 카메라 각재
            var gBm = G("beam", tG, 0, 290); A("beam", gBm, 0, 216, 52);
            Box(gBm, Mat(LAUAN), -15, 15, 186, 216, -60, 60, "lauan");
            // 12 카메라판
            var gCpl = G("cplate", tG, 40, 290); A("cplate", gCpl, 24, 282, 46);
            Box(gCpl, Mat(PLY), 15, 24, 186, 286, -50, 50, "plywood");
            foreach (var z in new[] { -25f, 25f }) CylX(gCpl, Mat(0x55595e), 3, 201, z, 24, 24.4f, "screw");
            // 13 HQ 카메라 + 렌즈 (개략)
            var gCam = G("camera", tG, 80, 290); A("camera", gCam, 62, 268, 0);
            foreach (var d in Quad(15, 15))
            {
                CylX(gCam, metal, 1.25f, CY + d.x, d.y, 15, 34, "M2.5");
                CylX(gCam, Mat(WHITE), 2.5f, CY + d.x, d.y, 24, 29, "spacer");
            }
            Box(gCam, Mat(GREEN), 29, 30.6f, CY - 19, CY + 19, -19, 19, "board");
            CylX(gCam, Mat(0x1d1f22, 0.5f), 17, CY, 0, 30.6f, 41, "lens_mount");
            CylX(gCam, Mat(0x15161a, 0.45f), 15, CY, 0, 41, 75, "lens");
            CylX(gCam, Metal(ALU), 15.3f, CY, 0, 58, 61, "lens_ring");
            CylX(gCam, Mat(0x0b0c0e, 0.2f, 0.6f), 9, CY, 0, 75, 75.3f, "lens_glass");
            camMount = Group("SemesCameraMount", tG);   // 광학 중심: 렌즈 앞면 뒤 2.2, 벽(+x)을 봄
            camMount.localPosition = P(75 - 2.2f, CY, 0);
            camMount.localRotation = RY(-Mathf.PI / 2);

            // 14 Θ 원점 스위치 (몸통·레버는 데이터시트 치수) + 감지돌기 STL 14
            var gHs = G("homesw", yG, 0, 130); A("homesw", gHs, -10, 170, -53);
            swT = KwSwitch(gHs, new Vector3(-9.565f, 157.451f, TH_LY), new Vector2(1, 0));
            var gDog = G("homesw", tG, 0, 290);
            gDog.name = "homesw_dog";
            Stl.Part(gDog, "14_theta_dog_v5.stl", Mat(PLA), s => new Vector3(s.x - 7.25f, 186 - s.z, -s.y - 45.75f));

            // 15 Y 끝 스위치: 브래킷 STL 15 v5 ×2 (−Y 쪽은 세로축으로 180° 돌림), 감지돌기 STL 15 v4
            var gYs = G("ysw", root); A("ysw", gYs, -58, 32, 22);
            Stl.Part(gYs, "15_y_switch_bracket_v5.stl", Mat(PLA), s => new Vector3(s.x - 63.9f, 9 + s.z, s.y + 25), "bracket_plusY");
            Stl.Part(gYs, "15_y_switch_bracket_v5.stl", Mat(PLA), s => new Vector3(-s.x - 42.1f, 9 + s.z, -s.y - 25), "bracket_minusY");
            foreach (var s in new[] { -1f, 1f }) swY.Add(KwSwitch(gYs, new Vector3(Y_LX, 18.451f, s * 19.681f), new Vector2(0, s)));
            var gYd = G("ysw", yG, 0, 70);
            gYd.name = "ysw_dog";
            Stl.Part(gYd, "15_y_dog_v4.stl", Mat(PLA), s => new Vector3(s.x - 53, 47.5f - s.z, -s.y));
            CylY(gYd, Metal(0x3b3f44), 2.75f, -55, 0, 56.5f, 59.5f, "bolt_head");

            // 16 제어 기판·전원 (개략)
            var gEl = G("elec", root); A("elec", gEl, -160, 26, 72);
            Action<float, float> stand = (x, z) => CylY(gEl, Mat(WHITE), 2.8f, x, z, 9, 15, "standoff");
            foreach (var xz in new[] { new Vector2(-200, 40), new Vector2(-128, 40), new Vector2(-200, 100), new Vector2(-128, 100) }) stand(xz.x, xz.y);
            Box(gEl, Mat(0xeceff1), -205, -122.5f, 15, 16.6f, 35, 105, "nucleo");
            Box(gEl, Mat(0x111111), -200, -128, 16.6f, 25, 37, 40); Box(gEl, Mat(0x111111), -200, -128, 16.6f, 25, 100, 103);
            Box(gEl, Mat(0x222222), -172, -158, 16.6f, 18, 63, 77);
            Box(gEl, Metal(ALU), -207, -198, 16.6f, 21, 66, 74);
            foreach (var xz in new[] { new Vector2(-196, -21), new Vector2(-124, -21), new Vector2(-196, 21), new Vector2(-124, 21) }) stand(xz.x, xz.y);
            Box(gEl, Mat(FR), -200, -120, 15, 16.6f, -25, 25, "perfboard_tmc2209");
            Box(gEl, Mat(0x6b3fa0), -175, -155, 16.6f, 18.2f, -8, 7); Box(gEl, Metal(ALU), -172, -158, 18.2f, 24, -5, 4);
            Box(gEl, Mat(0x2f8f4e), -198, -190, 16.6f, 26, -22, -12); Box(gEl, Mat(0x2f8f4e), -198, -190, 16.6f, 26, 12, 22);
            foreach (var xz in new[] { new Vector2(-196, -91), new Vector2(-124, -91), new Vector2(-196, -49), new Vector2(-124, -49) }) stand(xz.x, xz.y);
            Box(gEl, Mat(FR), -200, -120, 15, 16.6f, -95, -45, "perfboard_buffer");
            foreach (var z in new[] { -80f, -70f, -60f }) CylY(gEl, Mat(0x111111), 2.3f, -150, z, 16.6f, 21);
            Box(gEl, Mat(0x1f4fbf), -112, -69, 9, 10.6f, -112, -91, "lm2596"); CylY(gEl, Mat(0x111111), 6, -100, -101, 10.6f, 17); Box(gEl, Mat(0x3a7bd5), -80, -73, 10.6f, 16, -110, -104);
            Box(gEl, Mat(0xb8bcc0), -112, -94, 9, 17, 40, 53, "wago"); Box(gEl, Mat(0xf08a24), -112, -104, 17, 19, 40, 53);
            Box(gEl, Mat(0xb8bcc0), -112, -90, 9, 17, 60, 73, "wago"); Box(gEl, Mat(0xf08a24), -112, -104, 17, 19, 60, 73);
            Box(gEl, Mat(0x1b1c1f), -200, -140, 0, 32, 200, 320, "adapter_24V");

            // 18 라즈베리파이 5 + 홀더 (STL 18, Pi 는 상자)
            var gPi = G("pihold", yG, 0, 60); A("pihold", gPi, 8, 135, -64);
            Stl.Part(gPi, "18_rpi5_holder.stl", Mat(PLA), s => new Vector3(s.x + 6, 56.5f + s.z, s.y - 62.5f));
            {
                const float h = 56.5f;
                Box(gPi, Mat(GREEN), -35, 50, h + 14, h + 70, -64.6f, -63, "pi5_board");
                Box(gPi, Metal(ALU), -38, -14, h + 16, h + 68, -80.6f, -64.6f, "usb_eth");
                Box(gPi, Mat(0x2a2d31), -12, 30, h + 22, h + 62, -78.6f, -64.6f, "cooler");
                Box(gPi, Mat(0x111111), 33.8f, 43.8f, h + 70, h + 98, -70.6f, -64.6f, "usb_c");
            }

            // 카메라 시야 (SEMES 화각 verify_web 값: 플레이트 면 약 296 × 156)
            fovLines = MakeFov();

            foreach (var kv in parts) kv.Value.renderers.Clear();
            foreach (var tag in root.GetComponentsInChildren<PartTag>(true))
                foreach (var rd in tag.GetComponentsInChildren<Renderer>(true))
                    if (!owner.ContainsKey(rd)) { owner[rd] = tag.id; parts[tag.id].renderers.Add(rd); }
        }

        static IEnumerable<Vector2> Quad(float a, float b)
        {
            yield return new Vector2(-a, -b); yield return new Vector2(a, -b); yield return new Vector2(-a, b); yield return new Vector2(a, b);
        }

        // C클램프 (개략): 도면 코드와 같은 상자 3개 (x0, x1, y0, y1) + 나사
        void CClamp(Transform g, float z, float[] b1, float[] b2, float[] b3, float xs)
        {
            var red = Mat(RED); var metal = Metal(STEEL);
            foreach (var b in new[] { b1, b2, b3 }) Box(g, red, b[0], b[1], b[2], b[3], z - 6, z + 6, "clamp");
            CylY(g, metal, 4, xs, z, -92, -29, "screw");
            CylY(g, metal, 9, xs, z, -29, -25, "pad");
            CylZ(g, metal, 3.5f, xs, -88, z - 28, z + 28, "handle");
        }

        public static void Kill(UnityEngine.Object o)
        {
            if (Application.isPlaying) UnityEngine.Object.Destroy(o); else UnityEngine.Object.DestroyImmediate(o);
        }

        LineRenderer MakeFov()
        {
            var go = new GameObject("fov");
            go.transform.SetParent(tG, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;
            lr.widthMultiplier = 0.8f;
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.startColor = lr.endColor = new Color(0.11f, 0.37f, 0.82f, 0.7f);
            float L = 75 - 2.2f, R = QRX, d = R - L, hw = 1920f / SemesView.FX * d, hh = 1080f / SemesView.FY * d;   // 플레이트 면에서 화면 가장자리 (화각에서 계산)
            Vector3 a = P(L, CY, 0);
            var c = new[] { P(R, CY - hh, -hw), P(R, CY - hh, hw), P(R, CY + hh, hw), P(R, CY + hh, -hw) };
            var pts = new List<Vector3> { a, c[0], c[1], a, c[1], c[2], a, c[2], c[3], a, c[3], c[0] };
            lr.positionCount = pts.Count;
            lr.SetPositions(pts.ToArray());
            return lr;
        }

        // ---------- KW-10 스위치와 레버 (도면 코드 kwSw·leverPhi 그대로) ----------
        public class Sw { public Transform lever; public Vector3 o; public Vector2 d, lz; }
        const float KW_L = 12.8f, KW_H = 6.5f, KW_T = 5.8f, KW_RUN = 15.85f, KW_HU = 0.6f, KW_HV = 6.6f, KW_FREE = 12.2f;
        static readonly float TILT = 30 * Mathf.Deg2Rad, CT = Mathf.Cos(TILT), ST = Mathf.Sin(TILT);
        const float TH_LY = -50f, TH_DOGB = 174.0f, Y_LX = -53f, Y_DOGB = 35.0f;
        static readonly float PHI_FREE = Mathf.Asin((KW_FREE - KW_HV) / KW_RUN);
        Sw swT;
        readonly List<Sw> swY = new List<Sw>();

        Sw KwSwitch(Transform g, Vector3 o, Vector2 d)
        {
            var k = Group("KW-10", g);
            k.localPosition = P(o.x, o.y, o.z);
            float a = Mathf.Atan2(-d.y, d.x);
            k.localRotation = RY(a) * RZ(TILT);   // 도면 rotation.order 'YZX', (0, a, TILT)
            Box(k, Mat(0x202226), 0, KW_L, 0, KW_H, -KW_T / 2, KW_T / 2, "body");
            foreach (var u in new[] { 1.1f, 6.2f, 11.3f }) Box(k, Metal(STEEL), u - .45f, u + .45f, -4.3f, 0, -.25f, .25f, "pin");
            foreach (var u in new[] { 3.15f, 9.65f }) CylZ(k, Metal(0x3b3f44), 1.9f, u, 1.5f, -KW_T / 2 - 2, -KW_T / 2, "M2");
            Box(k, Mat(PLA), KW_L, KW_L + 1.5f, -3, 4, KW_T / 2 - 3, KW_T / 2, "angle_stop");
            var lv = Group("lever_pivot", k);
            lv.localPosition = P(KW_HU, KW_HV, 0);
            Box(lv, Metal(STEEL), 0, KW_RUN + .3f, -.15f, .15f, -1.2f, 1.2f, "lever");
            return new Sw { lever = lv, o = o, d = d, lz = new Vector2(Mathf.Sin(a), Mathf.Cos(a)) };
        }

        static Vector3 SwW(Sw sw, float u, float lat, float v)
        {
            float h = u * CT - v * ST, up = u * ST + v * CT;
            return new Vector3(sw.o.x + sw.d.x * h + sw.lz.x * lat, sw.o.y + up, sw.o.z + sw.d.y * h + sw.lz.y * lat);
        }

        static float LeverPhi(Sw sw, Func<float, float, float> bot)
        {
            bool Ok(float phi)
            {
                for (int i = 1; i <= 32; i++)
                {
                    float t = KW_RUN * i / 32f, u = KW_HU + t * Mathf.Cos(phi), v = KW_HV + t * Mathf.Sin(phi);
                    foreach (var lat in new[] { -1.2f, 0f, 1.2f })
                    {
                        var p = SwW(sw, u, lat, v);
                        if (p.y > bot(p.x, p.z) + 1e-6f) return false;
                    }
                }
                return true;
            }
            if (Ok(PHI_FREE)) return PHI_FREE;
            float lo = -0.4f, hi = PHI_FREE;
            for (int k = 0; k < 24; k++) { float m = (lo + hi) / 2; if (Ok(m)) lo = m; else hi = m; }
            return lo;
        }

        static float Fil(float d) => d >= .5f ? 0 : (d < 0 ? float.PositiveInfinity : .5f - Mathf.Sqrt(.25f - (.5f - d) * (.5f - d)));

        // 도면 자세 적용: ySim mm (가운데 0), thSim ° (+ = 위에서 볼 때 반시계, 도면 정의)
        public void SetPose(float ySim, float thSim)
        {
            yG.localPosition = P(0, 0, ySim);
            float f = -thSim * Mathf.Deg2Rad;            // 도면 tG.rotation.y
            tG.localRotation = RY(f);
            Func<float, float, float> yBot = (x, z) => (x >= -60 && x <= -46 && Mathf.Abs(z - ySim) <= 6) ? Y_DOGB + Fil(6 - Mathf.Abs(z - ySim)) : float.PositiveInfinity;
            foreach (var sw in swY) sw.lever.localRotation = RZ(LeverPhi(sw, yBot));
            float c = Mathf.Cos(f), sn = Mathf.Sin(f);
            Func<float, float, float> tBot = (x, z) =>
            {
                float xr = x * c - z * sn, zr = x * sn + z * c;
                return (xr >= -15 && xr <= 0.5f && zr >= TH_LY - 5.5f && zr <= TH_LY + 5.5f) ? TH_DOGB + Mathf.Max(Fil(0.5f - xr), Mathf.Max(-14 - xr, 0)) : float.PositiveInfinity;
            };
            swT.lever.localRotation = RZ(LeverPhi(swT, tBot));
        }

        // 보는 방향 (도면 VIEWS)
        public static readonly (string key, string label, Vector3 pos, Vector3 target)[] Views =
        {
            ("all", "전체", P(-560, 380, -480), P(40, 120, 0)),
            ("side", "옆 (Y 방향)", P(40, 175, -1000), P(40, 150, 0)),
            ("back", "사람 쪽", P(-950, 230, 0), P(60, 150, 0)),
            ("top", "위", P(83, 1350, 1), P(83, 0, 0)),
            ("mount", "출력 받침", P(-200, 200, -215), P(0, 92, -8)),
            ("cam", "카메라→QR", P(150, 360, -300), P(230, 240, 0)),
            ("target", "타깃", P(40, 520, -680), P(450, 230, 0)),
        };
    }

    public class PartTag : MonoBehaviour { public string id; }
}
