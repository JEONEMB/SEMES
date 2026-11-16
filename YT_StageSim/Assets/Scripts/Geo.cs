// 좌표 변환·기본 형상 (Y·Θ 검증 장비 시뮬레이터)
// 도면 좌표 = zip 의 index.html (three.js, 오른손 좌표, 단위 mm): x = 렌즈에서 벽 쪽, y = 책상 위 높이, z = Y 스테이지 방향
// Unity (왼손 좌표, 1 단위 = 1 mm): (x, y, z) → (x, y, -z). 회전은 거울 켤레: Rx(a) → Rx(-a), Ry(a) → Ry(-a), Rz(a) → Rz(a)
// 이렇게 옮기면 Unity 화면이 웹 도면 화면과 같게 보임
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace YT
{
    public static class Geo
    {
        public static Vector3 P(float x, float y, float z) => new Vector3(x, y, -z);
        public static Quaternion RX(float rad) => Quaternion.AngleAxis(-rad * Mathf.Rad2Deg, Vector3.right);
        public static Quaternion RY(float rad) => Quaternion.AngleAxis(-rad * Mathf.Rad2Deg, Vector3.up);
        public static Quaternion RZ(float rad) => Quaternion.AngleAxis(rad * Mathf.Rad2Deg, Vector3.forward);

        static readonly Dictionary<string, Material> Mats = new Dictionary<string, Material>();
        static Shader shader;

        public static Color Hex(uint c) => new Color(((c >> 16) & 255) / 255f, ((c >> 8) & 255) / 255f, (c & 255) / 255f);

        public static Material Mat(uint color, float rough = 0.72f, float metal = 0.05f)
        {
            string key = color.ToString("x6") + "_" + rough.ToString("0.00") + "_" + metal.ToString("0.00");
            if (Mats.TryGetValue(key, out var m)) return m;
            if (shader == null) shader = Shader.Find("Standard");
            m = new Material(shader) { color = Hex(color), name = key };
            m.SetFloat("_Glossiness", 1f - rough);
            m.SetFloat("_Metallic", metal);
            Mats[key] = m;
            return m;
        }

        public static Material Metal(uint color) => Mat(color, 0.4f, 0.45f);

        public static Transform Group(string name, Transform parent)
        {
            var g = new GameObject(name).transform;
            g.SetParent(parent, false);
            return g;
        }

        static GameObject Prim(PrimitiveType t, Transform parent, Material m, string name)
        {
            var go = GameObject.CreatePrimitive(t);
            go.name = name ?? t.ToString();
            go.transform.SetParent(parent, false);
            go.GetComponent<Renderer>().sharedMaterial = m;
            return go;
        }

        // 상자: 도면 좌표 범위
        public static Transform Box(Transform g, Material m, float x0, float x1, float y0, float y1, float z0, float z1, string name = null)
        {
            var go = Prim(PrimitiveType.Cube, g, m, name ?? "box");
            go.transform.localPosition = P((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
            go.transform.localScale = new Vector3(Mathf.Abs(x1 - x0), Mathf.Abs(y1 - y0), Mathf.Abs(z1 - z0));
            return go.transform;
        }

        // 원기둥 (Unity 원기둥: 높이 2, 반지름 0.5, y 축)
        public static Transform CylY(Transform g, Material m, float r, float x, float z, float y0, float y1, string name = null)
        {
            var go = Prim(PrimitiveType.Cylinder, g, m, name ?? "cyl");
            go.transform.localPosition = P(x, (y0 + y1) / 2, z);
            go.transform.localScale = new Vector3(2 * r, Mathf.Abs(y1 - y0) / 2, 2 * r);
            return go.transform;
        }

        public static Transform CylX(Transform g, Material m, float r, float y, float z, float x0, float x1, string name = null)
        {
            var go = Prim(PrimitiveType.Cylinder, g, m, name ?? "cyl");
            go.transform.localPosition = P((x0 + x1) / 2, y, z);
            go.transform.localRotation = RZ(Mathf.PI / 2);
            go.transform.localScale = new Vector3(2 * r, Mathf.Abs(x1 - x0) / 2, 2 * r);
            return go.transform;
        }

        public static Transform CylZ(Transform g, Material m, float r, float x, float y, float z0, float z1, string name = null)
        {
            var go = Prim(PrimitiveType.Cylinder, g, m, name ?? "cyl");
            go.transform.localPosition = P(x, y, (z0 + z1) / 2);
            go.transform.localRotation = RX(Mathf.PI / 2);
            go.transform.localScale = new Vector3(2 * r, Mathf.Abs(z1 - z0) / 2, 2 * r);
            return go.transform;
        }
    }

    // 이진 STL → Mesh. map: STL 좌표 → 도면 좌표. 결과는 Unity 좌표, 면 법선은 STL 저장 법선 방향에 맞춰 감는 방향을 정함
    public static class Stl
    {
        public static string Dir => Path.Combine(Application.streamingAssetsPath, "stl");

        public static double LoadMs, ColliderMs;   // 시작 시간 측정용

        public static Mesh Load(string file, Func<Vector3, Vector3> map)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            byte[] b = File.ReadAllBytes(Path.Combine(Dir, file));
            if (b.Length < 84) throw new Exception("STL 이 너무 짧음: " + file);
            int n = BitConverter.ToInt32(b, 80);
            if (84 + 50 * n != b.Length) throw new Exception("이진 STL 이 아님: " + file);
            var v = new Vector3[n * 3];
            var nr = new Vector3[n * 3];
            var idx = new int[n * 3];
            Vector3 o = map(Vector3.zero);
            for (int i = 0; i < n; i++)
            {
                int p = 84 + 50 * i;
                Vector3 sn = new Vector3(BitConverter.ToSingle(b, p), BitConverter.ToSingle(b, p + 4), BitConverter.ToSingle(b, p + 8));
                var t = new Vector3[3];
                for (int k = 0; k < 3; k++)
                {
                    int q = p + 12 + 12 * k;
                    Vector3 s = new Vector3(BitConverter.ToSingle(b, q), BitConverter.ToSingle(b, q + 4), BitConverter.ToSingle(b, q + 8));
                    Vector3 d = map(s);
                    t[k] = Geo.P(d.x, d.y, d.z);
                }
                Vector3 c = Vector3.Cross(t[1] - t[0], t[2] - t[0]);
                Vector3 want;
                if (sn.sqrMagnitude > 1e-12f) { Vector3 dd = map(sn) - o; want = Geo.P(dd.x, dd.y, dd.z); }
                else want = c;
                if (Vector3.Dot(c, want) < 0) { var tmp = t[1]; t[1] = t[2]; t[2] = tmp; c = -c; }
                Vector3 fn = c.sqrMagnitude > 0 ? c.normalized : Vector3.up;
                for (int k = 0; k < 3; k++) { v[3 * i + k] = t[k]; nr[3 * i + k] = fn; idx[3 * i + k] = 3 * i + k; }
            }
            var mesh = new Mesh { name = file };
            if (v.Length > 65000) mesh.indexFormat = IndexFormat.UInt32;
            mesh.vertices = v;
            mesh.normals = nr;
            mesh.triangles = idx;
            mesh.RecalculateBounds();
            LoadMs += sw.Elapsed.TotalMilliseconds;
            return mesh;
        }

        public static Transform Part(Transform parent, string file, Material m, Func<Vector3, Vector3> map, string name = null)
        {
            var go = new GameObject(name ?? file);
            go.transform.SetParent(parent, false);
            var mesh = Load(file, map);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = m;
            var sc = System.Diagnostics.Stopwatch.StartNew();
            if (mesh.vertexCount <= 60000) go.AddComponent<MeshCollider>().sharedMesh = mesh;   // 클릭 선택용
            else go.AddComponent<BoxCollider>();                                                // 큰 모형(모터)은 상자로
            ColliderMs += sc.Elapsed.TotalMilliseconds;
            return go.transform;
        }
    }
}
