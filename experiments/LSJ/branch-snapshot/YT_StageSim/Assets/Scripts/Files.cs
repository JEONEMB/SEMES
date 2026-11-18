// StreamingAssets 읽기 (STL, QR 그림)
//   Windows 실행 파일·편집기: 디스크에서 바로 읽음
//   웹 빌드: 브라우저는 파일을 바로 못 읽으므로 시작할 때 manifest.txt 에 적힌 파일을 모두 내려받아 두고 그것을 씀 (Preload)
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace YT
{
    public static class Files
    {
        static readonly Dictionary<string, byte[]> cache = new Dictionary<string, byte[]>();
        public static string Error;   // 내려받기 실패 이유 (웹)

        // rel: StreamingAssets 기준 상대경로 (예: "stl/09_harmonic_motor_psh42_v2.stl")
        public static byte[] Read(string rel)
        {
            if (cache.TryGetValue(rel, out var b)) return b;
            return File.ReadAllBytes(Path.Combine(Application.streamingAssetsPath, rel));
        }

        // progress: 0~1
        public static IEnumerator Preload(System.Action<float> progress = null)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            string root = Application.streamingAssetsPath.TrimEnd('/') + "/";
            string[] names;
            using (var r = UnityWebRequest.Get(root + "manifest.txt"))
            {
                yield return r.SendWebRequest();
                if (r.result != UnityWebRequest.Result.Success) { Error = "manifest.txt: " + r.error; yield break; }
                names = r.downloadHandler.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            }
            var reqs = new List<(string name, UnityWebRequestAsyncOperation op)>();
            foreach (var n in names) reqs.Add((n, UnityWebRequest.Get(root + n).SendWebRequest()));
            while (true)
            {
                int done = 0; foreach (var q in reqs) if (q.op.isDone) done++;
                progress?.Invoke(names.Length == 0 ? 1 : done / (float)names.Length);
                if (done == reqs.Count) break;
                yield return null;
            }
            foreach (var (name, op) in reqs)
            {
                var r = op.webRequest;
                if (r.result == UnityWebRequest.Result.Success) cache[name] = r.downloadHandler.data;
                else Error = name + ": " + r.error;
                r.Dispose();
            }
#else
            progress?.Invoke(1);
            yield break;
#endif
        }
    }
}
