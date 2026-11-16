// GitLab dev verify_web (알고리즘 동작 확인, 이영은 2026-11-12~13) 의 runner.py · algorithm.py 를 옮김
// 회차마다 흔들림 대기 → 촬영 → 추론 → 보정 이동, 최대 N회. 매 회차 '알고리즘 추론값' 과 '스테이지 실제 오차(STATUS − 센터)' 를 같이 기록
// 웹과 다른 점:
//  · 촬영: 장비 카메라 연결 전에는 시뮬레이터 가상 카메라(SEMES 화각) 화면을 3840×2160 흑백으로 찍어 넘김 (사용자 지시)
//  · 흔들림 판정: 시뮬레이터 화면은 흔들리지 않으므로 이동 끝을 확인하고 바로 '안정' 처리 (실제 카메라 연결 후 위상 상관 판정으로 바꿔야 함)
//  · 알고리즘: 실행기(align_worker) 경로가 있으면 같은 통신 규약(ALGN/ALGR)으로 띄우고, 없으면 verify_web 의 모의 추론(실제 오차 + 잡음)
//  · 알고리즘 고르기 (GitLab dev 11-13 00f96b7 · 11-16 algorithm.py AlgorithmChoices): 실행 파일 옆 algorithms.json 목록 + 모의 + 직접 지정(연결·설정 탭)
//  · 실행기 READY 줄의 기능(ref, roi)과 "OK … | 이름 x y w h cx cy; …" 위치(ROI)를 받아 화면 윤곽에 씀 (GitLab dev align_worker.cpp)
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace YT
{
    [Serializable]
    public class VerifySettings
    {
        // runner.py DEFAULTS 와 같은 이름·기본값
        public float center_y_mm = 0, center_t_deg = 0;
        public float tol_y_mm = 0.045f, tol_t_deg = 0.01f;
        public int max_trials = 20;
        public bool enable_t = true;
        public float max_step_y_mm = 2.0f, max_step_t_deg = 1.0f;
        public float random_y_mm = 1.5f, random_t_deg = 0.5f;
        public float approach_y_mm = 0.2f, approach_t_deg = 0.1f;
        public int average = 1;
        public float settle_px = 0.25f, settle_timeout_s = 6.0f;
        public bool save_images = true;
        public bool auto_reference = true;
        public float reference_max_age_min = 60f;
        // verify_web 실행 옵션 (__main__.py --algo-path, --algo-config, --algo-output, --algo-gray) — Unity 에서는 화면 설정에 둠
        public string algo_path = "", algo_config = "", algo_output = "offset";
        public bool algo_gray = false;
        // runner.py DEFAULTS algorithm: 고른 알고리즘 id. "mock" = 모의, "custom" = 연결·설정 탭에서 직접 지정, 그 밖 = algorithms.json 의 id
        // 빈 값 = 예전 설정 (직접 지정 경로가 있으면 custom, 없으면 mock)
        public string algorithm = "";

        // runner.py LIMITS
        public static readonly Dictionary<string, (double lo, double hi)> LIMITS = new Dictionary<string, (double, double)>
        {
            {"tol_y_mm", (0.001, 1)}, {"tol_t_deg", (0.0001, 1)}, {"max_trials", (1, 100)}, {"max_step_y_mm", (0.01, 10)},
            {"max_step_t_deg", (0.001, 5)}, {"random_y_mm", (0, 10)}, {"random_t_deg", (0, 5)}, {"approach_y_mm", (0, 2)}, {"approach_t_deg", (0, 1)},
            {"average", (1, 16)}, {"settle_px", (0.05, 5)}, {"settle_timeout_s", (0.5, 30)}, {"reference_max_age_min", (1, 1440)},
            {"center_y_mm", (-1000, 1000)}, {"center_t_deg", (-360, 360)},
        };

        public static VerifySettings Load(string path)
        {
            try { if (File.Exists(path)) return JsonUtility.FromJson<VerifySettings>(File.ReadAllText(path)) ?? new VerifySettings(); } catch { }
            return new VerifySettings();
        }

        public void Save(string path) { Directory.CreateDirectory(Path.GetDirectoryName(path)); File.WriteAllText(path, JsonUtility.ToJson(this, true)); }
        public VerifySettings Clone() => JsonUtility.FromJson<VerifySettings>(JsonUtility.ToJson(this));
    }

    public class RunError : Exception { public RunError(string m) : base(m) { } }
    public class Aborted : RunError { public Aborted() : base("정지 요청으로 멈췄습니다.") { } }

    public class VEntry
    {
        public string label; public int trial; public DateTime ts;
        public double rawY, rawT; public string output;
        public double offY, offT, cmdY, cmdT;
        public double? truY, truT, errY, errT;
        public bool within; public bool? truthWithin;
        public double inferMs, settleS; public bool settled; public double shiftPx;
        public List<Roi> rois = new List<Roi>();
    }

    // 알고리즘이 돌려준 위치 (algorithm.py parse_rois): 이름, 사각형 x y w h, 중심 cx cy (알고리즘 좌표 3840×2160)
    public class Roi
    {
        public string name; public int x, y, w, h; public double cx, cy;

        public static List<Roi> Parse(string text)
        {
            var o = new List<Roi>();
            if (string.IsNullOrEmpty(text)) return o;
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            foreach (var part in text.Split(';'))
            {
                var f = part.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (f.Length != 7) continue;
                try
                {
                    o.Add(new Roi { name = f[0], x = int.Parse(f[1], inv), y = int.Parse(f[2], inv), w = int.Parse(f[3], inv), h = int.Parse(f[4], inv),
                                    cx = double.Parse(f[5], inv), cy = double.Parse(f[6], inv) });
                }
                catch (FormatException) { }
            }
            return o;
        }
    }

    // 알고리즘 결과 한 번 (y, t 단위 mm, °)
    public class AlgoOut { public double y, t, ms; public List<Roi> rois = new List<Roi>(); }

    public class VRun
    {
        public string id; public DateTime started; public (double, double)? startOffset; public int trial, maxTrials;
        public string result, resultLabel, algorithm; public double centerY, centerT;
    }

    // ---------------------------------------------------------------- 알고리즘 (algorithm.py)
    public interface IAlgo
    {
        string Kind { get; }
        string Output { get; }
        string Detail { get; }
        string Error { get; }
        Task<AlgoOut> Infer(byte[] gray, int w, int h, (double, double)? truth);
        Task<AlgoOut> SetReference(byte[] gray, int w, int h);
        Task<bool> SupportsReference();   // 영점(센터 사진) 등록을 하나 (실행기 READY 줄의 'ref')
        void Close();
    }

    public class MockAlgo : IAlgo
    {
        readonly System.Random rng = new System.Random();
        public string Kind => "mock";
        public string Output => "offset";
        public string Detail => "모의 추론 (실제 오차 + 잡음)";
        public string Error => null;
        double Gauss(double sd) { double u1 = 1 - rng.NextDouble(), u2 = rng.NextDouble(); return sd * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }
        public Task<AlgoOut> Infer(byte[] g, int w, int h, (double, double)? truth)
        {
            if (truth == null) throw new RunError("모의 추론에는 스테이지 실제 오차가 필요함 (STATUS 실패?)");
            return Task.FromResult(new AlgoOut { y = truth.Value.Item1 + Gauss(0.008), t = truth.Value.Item2 + Gauss(0.002) });   // algorithm.py noise 0.008 mm, 0.002°
        }
        public Task<AlgoOut> SetReference(byte[] g, int w, int h) => Task.FromResult(new AlgoOut());
        public Task<bool> SupportsReference() => Task.FromResult(true);
        public void Close() { }
    }

    // native/align_worker 를 자식 프로세스로: "READY" → "ALGN"/"ALGR" + uint32 w + uint32 h + w*h 바이트 → "OK y t ms" / "ERR ..."
    public class WorkerAlgo : IAlgo
    {
        readonly string path, config, logPath; readonly bool gray; readonly double timeoutS = 30;
        Process proc; Stream stdin; StreamReader stdout;
        HashSet<string> caps;    // READY 줄에 알린 기능 (ref = 영점 등록, roi = 위치 돌려줌)
        public string Kind => "worker";
        public string Output { get; }
        public string Label;     // 목록에서 고른 이름 (성현 등). 없으면 파일 이름
        public string Detail => Label ?? Path.GetFileName(path);
        public string Error { get; private set; }

        public WorkerAlgo(string path, string output, bool gray, string config, string logPath = null)
        {
            this.logPath = logPath;
            if (!File.Exists(path)) throw new RunError($"알고리즘 실행기가 없음: {path}");
            if (output != "offset" && output != "command") throw new RunError("output 은 offset 또는 command");
            this.path = path; Output = output; this.gray = gray; this.config = config;
        }

        async Task Start()
        {
            var args = (gray ? "--gray " : "") + (string.IsNullOrEmpty(config) ? "" : $"--config \"{config}\"");
            var psi = new ProcessStartInfo(path, args)
            {
                UseShellExecute = false, RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = logPath != null,
                CreateNoWindow = true, WorkingDirectory = Path.GetDirectoryName(path), StandardOutputEncoding = Encoding.UTF8,
            };
            proc = Process.Start(psi);
            if (logPath != null)
            {   // 알고리즘이 표준 에러로 찍는 내용 (align_worker 는 표준 출력도 여기로 돌림)
                var lp = logPath;
                proc.ErrorDataReceived += (_, ev) => { if (ev.Data != null) try { File.AppendAllText(lp, ev.Data + Environment.NewLine); } catch { } };
                proc.BeginErrorReadLine();
            }
            stdin = proc.StandardInput.BaseStream; stdout = proc.StandardOutput;
            var line = await ReadLine(10);
            var words = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0 || words[0] != "READY") { Kill(); Error = line; throw new RunError($"알고리즘 실행기 시작 실패: {line}"); }
            caps = new HashSet<string>(words.Skip(1));
        }

        public async Task<bool> SupportsReference()
        {
            if (caps == null && proc == null)
            {
                try { await Start(); }
                catch (RunError e) { Error = e.Message; throw; }
            }
            return caps != null && caps.Contains("ref");
        }

        async Task<string> ReadLine(double timeout)
        {
            var t = stdout.ReadLineAsync();
            if (await Task.WhenAny(t, Task.Delay(TimeSpan.FromSeconds(timeout))) != t) throw new RunError($"알고리즘 응답 없음 ({timeout:0}s)");
            var s = t.Result;
            if (s == null) throw new RunError($"알고리즘 실행기가 종료됨 (코드 {(proc.HasExited ? proc.ExitCode.ToString() : "?")})");
            return s.Trim();
        }

        void Kill() { try { if (proc != null && !proc.HasExited) proc.Kill(); } catch { } proc = null; }

        async Task<string> Send(string magic, byte[] img, int w, int h)
        {
            if (proc != null && proc.HasExited) Kill();
            if (proc == null) await Start();
            try
            {
                await Task.Run(() =>
                {
                    stdin.Write(Encoding.ASCII.GetBytes(magic), 0, 4);
                    stdin.Write(BitConverter.GetBytes((uint)w), 0, 4); stdin.Write(BitConverter.GetBytes((uint)h), 0, 4);
                    stdin.Write(img, 0, img.Length); stdin.Flush();
                });
                return await ReadLine(timeoutS);
            }
            catch (Exception e) when (!(e is RunError)) { Kill(); Error = e.Message; throw new RunError(e.Message); }
            catch (RunError e) { Kill(); Error = e.Message; throw; }
        }

        // "OK y t ms[ | ROI...]"
        public async Task<AlgoOut> Infer(byte[] g, int w, int h, (double, double)? truth)
        {
            var line = await Send("ALGN", g, w, h);
            int bar = line.IndexOf(" | ", StringComparison.Ordinal);
            var p = (bar < 0 ? line : line.Substring(0, bar)).Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (p.Length == 4 && p[0] == "OK")
            {
                try { var o = new AlgoOut { y = D(p[1]), t = D(p[2]), ms = D(p[3]), rois = Roi.Parse(bar < 0 ? null : line.Substring(bar + 3)) }; Error = null; return o; }
                catch (FormatException) { }
            }
            Error = line; throw new RunError("알고리즘 오류: " + line);
        }

        // "OK 영점y 영점t[ | ROI...]"
        public async Task<AlgoOut> SetReference(byte[] g, int w, int h)
        {
            var line = await Send("ALGR", g, w, h);
            int bar = line.IndexOf(" | ", StringComparison.Ordinal);
            var p = (bar < 0 ? line : line.Substring(0, bar)).Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (p.Length >= 3 && p[0] == "OK")
            {
                try { return new AlgoOut { y = D(p[1]), t = D(p[2]), rois = Roi.Parse(bar < 0 ? null : line.Substring(bar + 3)) }; }
                catch (FormatException) { }
            }
            Error = line; throw new RunError("영점 등록 실패: " + line);
        }

        static double D(string s) => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture);
        public void Close() { try { stdin?.Close(); proc?.WaitForExit(3000); } catch { } Kill(); }
    }

    // ---------------------------------------------------------------- 알고리즘 목록 (algorithm.py AlgorithmChoices)
    // 실행 파일 옆 algorithms.json (verify_web/instance/algorithms.json 과 같은 모양). 상대경로 = 이 파일이 있는 폴더 기준
    [Serializable] public class AlgoEntry { public string id, label, path, config, output = "offset", desc; public bool gray; }
    [Serializable] class AlgoFile { public AlgoEntry[] algorithms; }

    public class AlgoList
    {
        public readonly string file;
        public List<AlgoEntry> entries = new List<AlgoEntry>();
        public string loadError;
        DateTime stamp; bool loaded;

        public AlgoList(string file) { this.file = file; Load(); }

        // 편집기 = 프로젝트 폴더, 빌드 = 실행 파일 폴더. 시험용으로 -ytalgos <파일> 로 바꿀 수 있음
        public static string DefaultFile()
        {
            var args = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(args, "-ytalgos");
            if (i >= 0 && i + 1 < args.Length) return Path.GetFullPath(args[i + 1]);
            return Path.Combine(Path.GetDirectoryName(Application.dataPath), "algorithms.json");
        }

        // 파일이 바뀌었으면 다시 읽음 (true = 바뀜)
        public bool Load()
        {
            var t = File.Exists(file) ? File.GetLastWriteTimeUtc(file) : DateTime.MinValue;
            if (loaded && t == stamp) return false;
            loaded = true; stamp = t; entries = new List<AlgoEntry>(); loadError = null;
            if (t == DateTime.MinValue) return true;
            try
            {
                var f = JsonUtility.FromJson<AlgoFile>(File.ReadAllText(file));
                var seen = new HashSet<string>();
                foreach (var e in f?.algorithms ?? new AlgoEntry[0])
                {
                    if (string.IsNullOrWhiteSpace(e.id) || e.id == "mock" || e.id == "custom" || !seen.Add(e.id)) { loadError = $"알고리즘 id 가 비었거나 겹침: {e.id}"; continue; }
                    if (string.IsNullOrWhiteSpace(e.label)) e.label = e.id;
                    if (e.output != "command") e.output = "offset";
                    entries.Add(e);
                }
            }
            catch (Exception e) { loadError = "algorithms.json 을 읽지 못함: " + e.Message; }
            return true;
        }

        public AlgoEntry Find(string id) => entries.FirstOrDefault(e => e.id == id);

        public string Resolve(string v)
        {
            if (string.IsNullOrWhiteSpace(v)) return null;
            v = v.Trim();
            if (v == "~" || v.StartsWith("~/") || v.StartsWith("~\\")) v = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile) + v.Substring(1);
            return Path.IsPathRooted(v) ? Path.GetFullPath(v) : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file), v));
        }

        // 실행기·설정 파일이 있나 (없으면 화면에 이유를 보여 줌)
        public string Missing(AlgoEntry e)
        {
            var p = Resolve(e.path);
            if (p == null || !File.Exists(p)) return $"실행기 없음: {p ?? "(path 비어 있음)"}";
            var c = Resolve(e.config);
            if (c != null && !File.Exists(c) && !Directory.Exists(c)) return $"설정·자료 없음: {c}";
            return null;
        }
    }

    // ---------------------------------------------------------------- 검증 루프 (runner.py Runner)
    public class VerifyRunner
    {
        public static readonly Dictionary<string, string> PHASES = new Dictionary<string, string>
        {
            {"idle", "대기"}, {"homing", "원점 복귀 중"}, {"positioning", "시작 위치로 이동 중"}, {"settling", "흔들림 대기"}, {"capturing", "촬영"},
            {"inferring", "추론"}, {"moving", "보정 이동 중"}, {"done", "완료"}, {"stopped", "정지됨"}, {"error", "오류"},
        };
        static readonly string[] DONE_RESULTS = { "정상", "정지 명령으로 중단", "리밋 스위치 감지", "드라이버 고장" };
        static readonly string[] STATE_NAMES = { "idle", "moving", "homing", "fault" };

        readonly App app;
        IStage stage => app.stage;
        public VerifySettings settings;
        readonly string settingsPath, runsDir;
        public IAlgo algo;
        public string algoInitError;
        public AlgoList algos;
        public string algoId = "mock";        // 지금 쓰는 알고리즘 (mock / custom / 목록 id)
        // 화면 윤곽 (verify_web state reference.rois · rois_now · aligned): 센터 기준 사진의 위치, 마지막 측정의 위치, 정렬 여부 (모르면 null)
        public List<Roi> refRois, roisNow; public bool? aligned; public bool refFixed;

        // 화면 상태
        public string phase = "idle", message = "", job;
        public bool busy, moving;
        public double? posY, posT; public int flags = -1; public string stageState, stageError;
        public VRun run; public VEntry last, measure; public readonly List<VEntry> history = new List<VEntry>();
        public (DateTime ts, double y, double t)? reference;
        public int version;
        bool abort;
        TaskCompletionSource<DoneInfo> doneWait;
        public Func<bool> OtherBusy;   // 시연·관리자 실행기 사용 중
        readonly System.Random rng = new System.Random();

        public VerifyRunner(App app, string dataDir)
        {
            this.app = app;
            settingsPath = Path.Combine(dataDir, "verify_settings.json");
            runsDir = Path.Combine(dataDir, "verify_runs");
            settings = VerifySettings.Load(settingsPath);
            onDone = d => doneWait?.TrySetResult(d);
            Attach(app.stage);
            algos = new AlgoList(AlgoList.DefaultFile());
            MakeAlgo();
        }

        public string RunsDir => runsDir;
        IStage attached; readonly Action<DoneInfo> onDone;
        // 스테이지를 바꿀 때 (모의 ↔ 실물) DONE 받기를 옮김
        public void Attach(IStage s) { if (attached != null) attached.Done -= onDone; attached = s; s.Done += onDone; Set(); }

        // 설정의 algorithm 으로 실행기를 만듦. 못 만들면 모의 추론 + 이유 표시
        public void MakeAlgo()
        {
            algo?.Close(); algoInitError = null;
            string id = settings.algorithm;
            if (string.IsNullOrEmpty(id)) id = string.IsNullOrWhiteSpace(settings.algo_path) ? "mock" : "custom";
            algoId = id;
            var log = Path.Combine(Path.GetDirectoryName(settingsPath), "align_worker.log");
            try
            {
                if (id == "custom")
                {
                    if (string.IsNullOrWhiteSpace(settings.algo_path)) throw new RunError("직접 지정한 실행기 경로가 비어 있음");
                    algo = new WorkerAlgo(settings.algo_path.Trim(), settings.algo_output, settings.algo_gray, settings.algo_config?.Trim(), log) { Label = "직접 지정 " + Path.GetFileName(settings.algo_path.Trim()) };
                    return;
                }
                if (id != "mock")
                {
                    var e = algos.Find(id) ?? throw new RunError($"algorithms.json 에 없는 알고리즘: {id}");
                    var why = algos.Missing(e);
                    if (why != null) throw new RunError($"{e.label}: {why}");
                    algo = new WorkerAlgo(algos.Resolve(e.path), e.output, e.gray, algos.Resolve(e.config), log) { Label = e.label };
                    return;
                }
            }
            catch (RunError e) { algoInitError = e.Message + " → 모의 추론 사용"; }
            algo = new MockAlgo();
        }

        // 목록에 보일 이름
        public string AlgoLabel(string id) => id == "mock" ? "모의" : id == "custom" ? "직접 지정" : algos.Find(id)?.label ?? id;

        // runner.py select_algorithm: 다른 알고리즘으로 바꿈. 영점·기록이 알고리즘마다 달라서 센터 기준과 지난 기록은 지움
        public void SelectAlgorithm(string id)
        {
            if (busy) throw new RunError("작업 중에는 알고리즘을 바꿀 수 없습니다.");
            if (OtherBusy != null && OtherBusy()) throw new RunError("다른 화면에서 장비를 쓰는 중입니다.");
            if (id == "custom" && string.IsNullOrWhiteSpace(settings.algo_path)) throw new RunError("연결·설정 탭 ③ 에서 실행기 경로를 먼저 넣으세요.");
            if (id != "mock" && id != "custom")
            {
                var e = algos.Find(id) ?? throw new RunError($"알 수 없는 알고리즘: {id}");
                var why = algos.Missing(e);
                if (why != null) throw new RunError($"{e.label}: {why}");
            }
            var n = settings.Clone(); n.algorithm = id;
            settings = n; settings.Save(settingsPath);
            MakeAlgo();
            if (algoInitError != null) throw new RunError(algoInitError);
            Set(() =>
            {
                reference = null; refRois = null; roisNow = null; aligned = null; refFixed = false;
                run = null; history.Clear(); last = null; measure = null; phase = "idle";
                message = $"알고리즘: {AlgoLabel(algoId)}. 센터 기준은 다음 시작 때 새로 잡습니다.";
            });
        }

        void Set(Action f = null) { f?.Invoke(); version++; }
        void Phase(string p, string msg = "") { CheckAbort(); Set(() => { phase = p; message = msg; }); }
        void CheckAbort() { if (abort) throw new Aborted(); }

        // runner.py Settings.update: 이름·범위 검사 후 저장
        public void UpdateSettings(VerifySettings n)
        {
            foreach (var kv in VerifySettings.LIMITS)
            {
                var f = typeof(VerifySettings).GetField(kv.Key);
                double v = Convert.ToDouble(f.GetValue(n));
                if (double.IsNaN(v) || double.IsInfinity(v)) throw new RunError($"{kv.Key}: 숫자가 필요함");
                if (v < kv.Value.lo || v > kv.Value.hi) throw new RunError($"{kv.Key}: {kv.Value.lo} ~ {kv.Value.hi}");
            }
            if (n.algo_output != "offset" && n.algo_output != "command") throw new RunError("algo_output: offset 또는 command");
            bool algoChanged = n.algo_path != settings.algo_path || n.algo_config != settings.algo_config || n.algo_output != settings.algo_output || n.algo_gray != settings.algo_gray || n.algorithm != settings.algorithm;
            settings = n.Clone(); settings.Save(settingsPath);
            if (algoChanged) MakeAlgo();
            Set();
        }

        // ---------------- 작업 실행
        // 검사는 호출한 자리에서 예외로 (화면 알림), 작업은 그 뒤 비동기로
        void Launch(string name, Func<System.Threading.Tasks.Task> fn)
        {
            if (busy) throw new RunError("다른 작업이 진행 중입니다. 정지 후 다시 시도하세요.");
            if (OtherBusy != null && OtherBusy()) throw new RunError("시연·관리자 화면에서 장비를 쓰는 중입니다. 그쪽 실행을 끝내거나 정지하세요.");
            abort = false; Set(() => { busy = true; job = name; message = ""; });
            Guard(fn);
        }

        async void Guard(Func<System.Threading.Tasks.Task> fn)
        {
            try { await fn(); }
            catch (Aborted) { Set(() => { phase = "stopped"; message = "정지 요청으로 멈췄습니다."; }); }
            catch (Exception e)
            {
                if (!(e is RunError)) UnityEngine.Debug.LogException(e);
                Set(() => { phase = "error"; message = e.Message; });
                if (run != null && run.result == null) FinishRun("error", e.Message);
            }
            finally { Set(() => { busy = false; job = null; moving = false; }); RefreshStatus(); }
        }

        public void Home() => Launch("home", () => HomeJob(1));
        public void ThetaZero() => Launch("home_t", () => HomeJob(2));   // Θ 는 원점 스위치 없음: HOME mask 2 = 지금 자리를 0 으로
        public void Goto(object offset) => Launch("goto", () => GotoJob(offset));
        public void Run() { CamCheck(); Launch("run", () => RunJob(null)); }
        public void Step() { CamCheck(); Launch("step", StepJob); }
        public void Measure() { CamCheck(); Launch("measure", MeasureJob); }
        public void Reference() { CamCheck(); Launch("reference", ReferenceJob); }

        // 3D 시뮬레이터는 실물 장비 동작에 절대 관여하지 않음: 실물 스테이지는 장비 카메라 영상으로만 보정
        void CamCheck()
        {
            if (stage != app.mock && app.cam.Kind == "sim")
                throw new RunError("실물 스테이지는 장비 카메라로만 보정합니다. 연결·설정 탭에서 카메라를 '라즈베리파이 영상'으로 연결하세요.");
        }

        public void Stop()
        {
            abort = true;
            if (moving) { try { stage.Stop(); } catch (Exception e) { Set(() => message = "STOP 실패: " + e.Message); } }
        }

        public void SetCenterHere()
        {
            if (busy) throw new RunError("작업 중에는 센터를 바꿀 수 없습니다.");
            var p = RefreshStatus();
            if (p == null) throw new RunError("스테이지 위치를 읽지 못했습니다.");
            var n = settings.Clone(); n.center_y_mm = p.Value.Item1 / 1000f; n.center_t_deg = p.Value.Item2 / 1000f;
            UpdateSettings(n);
            Set(() => message = $"센터 = Y {p.Value.Item1 / 1000.0:0.000} mm, Θ {p.Value.Item2 / 1000.0:0.000}°");
        }

        // ---------------- 스테이지
        public (int, int)? RefreshStatus()
        {
            try
            {
                var s = stage.Status;
                posY = s.yUm / 1000.0; posT = s.tMdeg / 1000.0; flags = (int)s.flags; stageState = STATE_NAMES[(int)s.state]; stageError = null;
                version++;
                return (s.yUm, s.tMdeg);
            }
            catch (Exception e) { stageError = e.Message; version++; return null; }
        }

        public void Poll() { if (!busy) RefreshStatus(); }

        (double, double)? Truth()
        {
            var p = RefreshStatus();
            if (p == null) return null;
            return (p.Value.Item1 / 1000.0 - settings.center_y_mm, p.Value.Item2 / 1000.0 - settings.center_t_deg);
        }

        static int Um(double mm) => (int)Math.Round(mm * 1000);

        async System.Threading.Tasks.Task Move(int yUm, int tMdeg, bool relative, string ph)
        {
            Phase(ph);
            if (!settings.enable_t && relative) tMdeg = 0;
            var lim = stage.Config;
            if (!relative && !(lim.yMinUm <= yUm && yUm <= lim.yMaxUm && lim.tMinMdeg <= tMdeg && tMdeg <= lim.tMaxMdeg))
                throw new RunError($"목표가 스테이지 한계 밖: Y {yUm / 1000.0:0.000} mm (한계 {lim.yMinUm / 1000.0:0.000}~{lim.yMaxUm / 1000.0:0.000})");
            Set(() => { moving = true; aligned = null; roisNow = null; });   // 움직이면 다시 재기 전까지 정렬 여부를 모름
            DoneInfo done;
            try
            {
                doneWait = new TaskCompletionSource<DoneInfo>();
                var r = relative ? stage.MoveRel(yUm, tMdeg) : stage.MoveAbs(yUm, tMdeg);
                if (r != NakReason.None) throw new RunError("STM32 명령 거부: " + StText.Nak(r));
                done = await WaitDone(60000);
            }
            finally { Set(() => moving = false); }
            Set(() => { posY = done.yUm / 1000.0; posT = done.tMdeg / 1000.0; });
            CheckAbort();
            if (done.result != DoneResult.Ok) throw new RunError("이동 실패: " + DONE_RESULTS[(int)done.result]);
        }

        // verify_web runner: wait_done(timeout_ms=60000 / 130000). 시간이 지나면 상태만 조회하고 이동 명령은 다시 보내지 않음
        // 시간은 실제 시간 기준 (모의 장비 배속과 무관)
        async System.Threading.Tasks.Task<DoneInfo> WaitDone(int timeoutMs)
        {
            var t = doneWait.Task;
            if (await System.Threading.Tasks.Task.WhenAny(t, System.Threading.Tasks.Task.Delay(timeoutMs)) == t) return t.Result;
            RefreshStatus();
            throw new RunError("STM32 이동 완료 응답 시간 초과. 이동 명령은 재전송하지 않았습니다.");
        }

        async System.Threading.Tasks.Task HomeJob(int mask)
        {
            Phase("homing");
            Set(() => { moving = true; aligned = null; roisNow = null; });
            DoneInfo done;
            try
            {
                doneWait = new TaskCompletionSource<DoneInfo>();
                var r = stage.Home(mask);
                if (r != NakReason.None) throw new RunError("STM32 명령 거부: " + StText.Nak(r));
                done = await WaitDone(130000);
            }
            finally { Set(() => moving = false); }
            CheckAbort();
            if (done.result != DoneResult.Ok) throw new RunError((mask == 1 ? "원점 복귀 실패: " : "Θ 원점 설정 실패: ") + DONE_RESULTS[(int)done.result]);
            Set(() => { phase = "idle"; message = mask == 1 ? "Y 원점 복귀 완료" : "Θ 원점 설정 완료 (지금 자리 = 0°)"; });
        }

        (double, double) ResolveOffset(object offset)
        {
            if (offset is string s && s == "random")
            {
                double ry = settings.random_y_mm, rt = settings.random_t_deg;
                double y = (rng.NextDouble() * 2 - 1) * ry;
                if (Math.Abs(y) < ry * 0.2) y = (y < 0 ? -1 : 1) * ry * 0.2;   // 너무 가까운 시작은 조금 띄움
                double t = settings.enable_t ? (rng.NextDouble() * 2 - 1) * rt : 0;
                return (Math.Round(y, 3), Math.Round(t, 3));
            }
            var o = ((double, double))offset;
            return (o.Item1, settings.enable_t ? o.Item2 : 0);
        }

        // 항상 −쪽 접근점을 거쳐 +방향으로 마지막 접근 (백래시 방향 통일)
        async System.Threading.Tasks.Task MoveTo(double yMm, double tDeg, string ph)
        {
            if (!settings.enable_t) tDeg = posT ?? 0;
            double ay = settings.approach_y_mm, at = settings.enable_t ? settings.approach_t_deg : 0;
            if (ay > 0 || at > 0) await Move(Um(yMm - ay), Um(tDeg - at), false, ph);
            await Move(Um(yMm), Um(tDeg), false, ph);
        }

        async System.Threading.Tasks.Task<(double, double)> GotoOffset(object offset)
        {
            var (y, t) = ResolveOffset(offset);
            await MoveTo(settings.center_y_mm + y, settings.center_t_deg + t, "positioning");
            return (y, t);
        }

        async System.Threading.Tasks.Task GotoJob(object offset)
        {
            var (y, t) = await GotoOffset(offset);
            Set(() => { phase = "idle"; message = $"시작 위치 도착: 센터 + Y {y:+0.000;-0.000} mm, Θ {t:+0.000;-0.000}°"; });
        }

        // ---------------- 검증
        VRun NewRun((double, double)? start)
        {
            var r = new VRun { id = DateTime.Now.ToString("yyyyMMdd_HHmmss"), started = DateTime.Now, startOffset = start, maxTrials = settings.max_trials,
                algorithm = algoId, centerY = settings.center_y_mm, centerT = settings.center_t_deg };
            Set(() => { run = r; history.Clear(); last = null; });
            return r;
        }

        void FinishRun(string result, string label)
        {
            if (run == null) return;
            run.result = result; run.resultLabel = label; version++;
            try
            {
                var d = Path.Combine(runsDir, run.id); Directory.CreateDirectory(d);
                var sb = new StringBuilder();
                sb.AppendLine("{");
                sb.AppendLine($"  \"run\": {{\"id\": \"{run.id}\", \"started\": \"{run.started:yyyy-MM-ddTHH:mm:ss}\", \"trial\": {run.trial}, \"max_trials\": {run.maxTrials}, \"result\": \"{result}\", \"result_label\": \"{label.Replace("\"", "'")}\", \"algorithm\": \"{run.algorithm}\", \"center\": {{\"y_mm\": {F(run.centerY)}, \"t_deg\": {F(run.centerT)}}}}},");
                sb.AppendLine($"  \"settings\": {JsonUtility.ToJson(settings)},");
                sb.AppendLine("  \"history\": [" + string.Join(",", history.Select(EntryJson)) + "]");
                sb.AppendLine("}");
                File.WriteAllText(Path.Combine(d, "run.json"), sb.ToString());
            }
            catch (Exception e) { UnityEngine.Debug.LogException(e); }
        }

        // verify_web run.json history 항목과 같은 이름
        static string EntryJson(VEntry e)
        {
            string N(double? v) => v == null ? "null" : F(v.Value);
            string B(bool? v) => v == null ? "null" : v.Value ? "true" : "false";
            return $"{{\"label\": \"{e.label}\", \"trial\": {e.trial}, \"ts\": \"{e.ts:yyyy-MM-ddTHH:mm:ss.fff}\", \"raw\": {{\"y\": {F(e.rawY)}, \"t\": {F(e.rawT)}}}, \"output\": \"{e.output}\", " +
                   $"\"offset\": {{\"y_mm\": {F(e.offY)}, \"t_deg\": {F(e.offT)}}}, \"command\": {{\"y_mm\": {F(e.cmdY)}, \"t_deg\": {F(e.cmdT)}}}, " +
                   $"\"truth\": {(e.truY == null ? "null" : $"{{\"y_mm\": {N(e.truY)}, \"t_deg\": {N(e.truT)}}}")}, \"error\": {(e.errY == null ? "null" : $"{{\"y_mm\": {N(e.errY)}, \"t_deg\": {N(e.errT)}}}")}, " +
                   $"\"within\": {B(e.within)}, \"truth_within\": {B(e.truthWithin)}, \"infer_ms\": {F(e.inferMs)}, \"settle_s\": {F(e.settleS)}, \"settled\": {B(e.settled)}, \"shift_px\": {F(e.shiftPx)}}}";
        }

        static string F(double v) => v.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture);

        bool ReferenceNeeded()
        {
            if (!settings.auto_reference) return false;
            return reference == null || (DateTime.Now - reference.Value.ts).TotalMinutes > settings.reference_max_age_min;
        }

        async System.Threading.Tasks.Task EnsureReference(bool comeBack)
        {
            if (!ReferenceNeeded()) return;
            var pos = comeBack ? RefreshStatus() : null;
            var old = reference;
            Set(() => message = "센터 기준이 " + (old == null ? "없어" : "오래돼") + " 먼저 잡습니다.");
            await ReferenceJob();
            if (pos != null) await MoveTo(pos.Value.Item1 / 1000.0, pos.Value.Item2 / 1000.0, "positioning");
        }

        async System.Threading.Tasks.Task RunJob(object offset)
        {
            await EnsureReference(offset == null);
            (double, double)? start = null;
            if (offset != null) start = await GotoOffset(offset);
            var r = NewRun(start);
            for (int n = 1; n <= r.maxTrials; n++)
            {
                if (await Trial(n, true))
                {
                    FinishRun("converged", $"{n}회차에 허용 오차 안으로 들어옴");
                    Set(() => { phase = "done"; message = run.resultLabel; });
                    return;
                }
            }
            FinishRun("max_trials", $"{r.maxTrials}회 안에 수렴하지 못함");
            Set(() => { phase = "done"; message = run.resultLabel; });
        }

        async System.Threading.Tasks.Task StepJob()
        {
            if (run == null || run.result != null) { await EnsureReference(true); NewRun(null); }
            int n = run.trial + 1;
            if (await Trial(n, true)) { FinishRun("converged", $"{n}회차에 허용 오차 안으로 들어옴"); Set(() => { phase = "done"; message = run.resultLabel; }); }
            else Set(() => { phase = "idle"; message = $"{n}회차 완료 (한 단계)"; });
        }

        async System.Threading.Tasks.Task ReferenceJob()
        {
            await MoveTo(settings.center_y_mm, settings.center_t_deg, "positioning");
            Phase("settling"); await Settle();
            Phase("capturing");
            var (img, w, h) = await Capture(Math.Max(4, settings.average));   // verify_web: 기준은 4장 이상 평균
            bool fixedZero = !await algo.SupportsReference();
            AlgoOut z;
            if (fixedZero) z = new AlgoOut();
            else { Phase("inferring"); z = await algo.SetReference(img, w, h); }
            { Directory.CreateDirectory(runsDir); Bmp.WriteGray(Path.Combine(runsDir, $"reference_{DateTime.Now:yyyyMMdd_HHmmss}.bmp"), img, w, h); }
            Set(() =>
            {
                reference = (DateTime.Now, z.y, z.t); refRois = z.rois; refFixed = fixedZero; phase = "idle";
                if (fixedZero) { aligned = null; roisNow = null; message = "센터 사진을 화면용으로만 찍었습니다 (이 알고리즘은 자료에 고정된 영점을 씀)."; }
                else { aligned = true; roisNow = z.rois; message = $"센터 기준 등록: 학습 영점 대비 Y {z.y * 1000:+0.0;-0.0} µm, Θ {z.t:+0.0000;-0.0000}°"; }
            });
        }

        async System.Threading.Tasks.Task MeasureJob()
        {
            await EnsureReference(true);
            var e = await Observe("측정");
            Set(() => { measure = e.Item1; phase = "idle"; message = "측정만 했습니다 (이동 없음)."; });
        }

        // 흔들림 판정·촬영은 카메라 쪽 (시뮬레이터: 이동 끝나면 안정 / 파이: verify_web wait_settled·Rectifier 그대로)
        System.Threading.Tasks.Task<(bool, double, double)> Settle() { CamCheck(); return app.cam.Settle(settings.settle_px, settings.settle_timeout_s, () => abort); }
        System.Threading.Tasks.Task<(byte[], int, int)> Capture(int average) { CamCheck(); return app.cam.Grab(average); }

        async System.Threading.Tasks.Task<(VEntry, byte[])> Observe(string label)
        {
            var s = settings;
            Phase("settling");
            var (settled, settleS, shift) = await Settle();
            Phase("capturing");
            byte[] img = null; int w = 0, h = 0;
            if (algo.Kind != "mock" || s.save_images) (img, w, h) = await Capture(s.average);
            var truth = Truth();
            Phase("inferring");
            var r = await algo.Infer(img, w, h, truth);
            string outp = algo.Output;
            Set(() => roisNow = r.rois);
            double oy = outp == "offset" ? r.y : -r.y, ot = outp == "offset" ? r.t : -r.t;
            double cy = -oy, ct = -ot;
            bool within = Math.Abs(oy) <= s.tol_y_mm && (!s.enable_t || Math.Abs(ot) <= s.tol_t_deg);
            var e = new VEntry
            {
                label = label, ts = DateTime.Now, rawY = r.y, rawT = r.t, output = outp, offY = oy, offT = ot, cmdY = cy, cmdT = s.enable_t ? ct : 0,
                truY = truth?.Item1, truT = truth?.Item2, errY = truth == null ? (double?)null : oy - truth.Value.Item1, errT = truth == null ? (double?)null : ot - truth.Value.Item2,
                within = within, truthWithin = truth == null ? (bool?)null : Math.Abs(truth.Value.Item1) <= s.tol_y_mm && (!s.enable_t || Math.Abs(truth.Value.Item2) <= s.tol_t_deg),
                inferMs = Math.Round(r.ms, 1), settleS = Math.Round(settleS, 2), settled = settled, shiftPx = shift, rois = r.rois,
            };
            Set(() => aligned = within);
            return (e, img);
        }

        async System.Threading.Tasks.Task<bool> Trial(int n, bool move)
        {
            var s = settings;
            Set(() => run.trial = n);
            var (e, img) = await Observe($"{n}회차");
            e.trial = n;
            if (s.save_images && img != null)
            {
                var d = Path.Combine(runsDir, run.id); Directory.CreateDirectory(d);
                Bmp.WriteGray(Path.Combine(d, $"trial_{n:00}.bmp"), img, 3840, 2160);
                AppendCsv(Path.Combine(d, "trials.csv"), e);
            }
            Set(() => { history.Add(e); if (history.Count > 100) history.RemoveAt(0); last = e; });
            if (e.within || !move) return e.within;
            double cy = e.cmdY, ct = e.cmdT;
            if (Math.Abs(cy) > s.max_step_y_mm || Math.Abs(ct) > s.max_step_t_deg)
                throw new RunError($"추론 명령이 한도를 넘어 멈춤: Y {cy:+0.000;-0.000} mm (한도 ±{s.max_step_y_mm}), Θ {ct:+0.000;-0.000}° (한도 ±{s.max_step_t_deg})");
            if (posY == null) await Move(Um(cy), Um(ct), true, "moving");
            else await MoveTo(posY.Value + cy, posT.Value + ct, "moving");
            return false;
        }

        static void AppendCsv(string path, VEntry e)
        {
            bool fresh = !File.Exists(path);
            string N(double? v) => v == null ? "" : F(v.Value);
            using (var w = new StreamWriter(path, true, new UTF8Encoding(false)))
            {
                if (fresh) w.WriteLine("trial,time,raw_y,raw_t,offset_y_mm,offset_t_deg,cmd_y_mm,cmd_t_deg,truth_y_mm,truth_t_deg,err_y_mm,err_t_deg,within,truth_within,infer_ms,settle_s,settled,shift_px");
                w.WriteLine(string.Join(",", e.trial, e.ts.ToString("HH:mm:ss"), F(e.rawY), F(e.rawT), F(e.offY), F(e.offT), F(e.cmdY), F(e.cmdT),
                    N(e.truY), N(e.truT), N(e.errY), N(e.errT), e.within, e.truthWithin?.ToString() ?? "", F(e.inferMs), F(e.settleS), e.settled, F(e.shiftPx)));
            }
        }

        // verify_web devices.py: 장치를 잡을 때 Y 속도 1 mm/s (STM32 리셋 뒤 기본 0.1 mm/s)
        public void Claim()
        {
            if (stage.Status.state == StState.Idle && stage.SetSpeed(1000, 0) == NakReason.None) Set(() => message = "장치를 잡으며 Y 속도를 1 mm/s 로 맞췄습니다.");
        }
    }

    // 8bit 흑백 BMP (verify_web 이 cv2.imwrite 로 남기는 trial_NN.bmp 와 같은 형식)
    public static class Bmp
    {
        public static void WriteGray(string path, byte[] g, int w, int h)
        {
            int stride = (w + 3) & ~3, img = stride * h, off = 14 + 40 + 1024;
            using (var f = new BinaryWriter(File.Create(path)))
            {
                f.Write((byte)'B'); f.Write((byte)'M'); f.Write(off + img); f.Write(0); f.Write(off);
                f.Write(40); f.Write(w); f.Write(h); f.Write((short)1); f.Write((short)8); f.Write(0); f.Write(img); f.Write(2835); f.Write(2835); f.Write(256); f.Write(0);
                for (int i = 0; i < 256; i++) { f.Write((byte)i); f.Write((byte)i); f.Write((byte)i); f.Write((byte)0); }
                var pad = new byte[stride - w];
                for (int y = h - 1; y >= 0; y--) { f.Write(g, y * w, w); if (pad.Length > 0) f.Write(pad); }
            }
        }
    }
}
