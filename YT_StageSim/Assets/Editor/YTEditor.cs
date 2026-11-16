// 편집기 메뉴: 장면 만들기, 화면 캡처 (명령줄: -executeMethod YT.YTEditor.CaptureCli)
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace YT
{
    public static class YTEditor
    {
        const string ScenePath = "Assets/Scenes/Main.unity";

        [MenuItem("YT/장면 만들기 (Main)")]
        public static void MakeScene()
        {
            Directory.CreateDirectory("Assets/Scenes");
            var sc = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("App").AddComponent<App>();
            EditorSceneManager.SaveScene(sc, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            Debug.Log("장면 저장: " + ScenePath);
        }

        // 편집 모드에서 장비를 만들어 보는 방향마다 PNG 저장 (확인용)
        [MenuItem("YT/화면 캡처 (Captures 폴더)")]
        public static void Capture()
        {
            var root = new GameObject("CaptureRoot").transform;
            try
            {
                App.BuildWorld(root, out var rig, out var cam, out var orbit, out var semes);
                Directory.CreateDirectory("Captures");
                var rt = new RenderTexture(1600, 900, 24) { antiAliasing = 4 };
                foreach (var v in Rig.Views)
                {
                    cam.transform.position = v.pos; cam.transform.LookAt(v.target);
                    Save(cam, rt, "Captures/view_" + v.key + ".png");
                }
                foreach (var pose in new[] { new Vector2(0, 0), new Vector2(20, 0), new Vector2(0, 8) })
                {
                    rig.SetPose(pose.x, pose.y);
                    Save(semes.cam, semes.rt, $"Captures/semes_Y{pose.x:0}_T{pose.y:0}.png");
                    var qr = semes.Px(rig.root.TransformPoint(Geo.P(Rig.QRX, Rig.CY, 0)));
                    Debug.Log($"YT 캡처: 자세 Y {pose.x} mm, Θ {pose.y}° → QR 중심 {qr.x:0.0}, {qr.y:0.0} px");
                }
                Debug.Log("YT 캡처 끝: " + Path.GetFullPath("Captures"));
            }
            finally { Object.DestroyImmediate(root.gameObject); }
        }

        static void Save(Camera cam, RenderTexture rt, string path)
        {
            var old = cam.targetTexture;
            cam.targetTexture = rt;
            if (cam.name == "SemesCamera") cam.aspect = (1920f / SemesView.FX) / (1080f / SemesView.FY);
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            RenderTexture.active = null; cam.targetTexture = old;
            Object.DestroyImmediate(tex);
        }

        public static void CaptureCli()
        {
            MakeScene();
            Capture();
            EditorApplication.Exit(0);
        }
        // 실행 파일 빌드 (명령줄: -executeMethod YT.YTEditor.BuildCli) → Build/YT_StageSim.exe
        [MenuItem("YT/실행 파일 빌드 (Windows)")]
        public static void Build()
        {
            MakeScene();
            PlayerSettings.productName = "YT_StageSim";
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1600; PlayerSettings.defaultScreenHeight = 900;
            PlayerSettings.resizableWindow = true;
            IncludeShaders("Standard", "Unlit/Transparent", "Sprites/Default", "UI/Default", "Legacy Shaders/Transparent/Diffuse");
            var r = BuildPipeline.BuildPlayer(new[] { ScenePath }, "Build/YT_StageSim.exe", BuildTarget.StandaloneWindows64, BuildOptions.None);
            Debug.Log("YT 빌드 결과: " + r.summary.result + ", 오류 " + r.summary.totalErrors);
            // 알고리즘 목록 (실행 파일 옆 algorithms.json, 위 막대에서 고름)
            if (r.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded && System.IO.File.Exists("algorithms.json"))
                System.IO.File.Copy("algorithms.json", "Build/algorithms.json", true);
        }

        // 코드에서 Shader.Find 로 찾는 셰이더는 빌드에서 빠지므로 '항상 포함' 목록에 넣음
        static void IncludeShaders(params string[] names)
        {
            var gs = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>("ProjectSettings/GraphicsSettings.asset");
            var so = new SerializedObject(gs);
            var arr = so.FindProperty("m_AlwaysIncludedShaders");
            foreach (var n in names)
            {
                var sh = Shader.Find(n);
                if (sh == null) { Debug.LogWarning("셰이더 없음: " + n); continue; }
                bool has = false;
                for (int i = 0; i < arr.arraySize; i++) if (arr.GetArrayElementAtIndex(i).objectReferenceValue == sh) has = true;
                if (has) continue;
                arr.InsertArrayElementAtIndex(arr.arraySize);
                arr.GetArrayElementAtIndex(arr.arraySize - 1).objectReferenceValue = sh;
            }
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
        }

        public static void BuildCli()
        {
            Build();
            EditorApplication.Exit(0);
        }
    }
}
