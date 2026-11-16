// 실제 STM32 연결 (GitLab dev STM32/robot_stage_protocol.md, demo_web/serial_stage.py 를 옮김)
//   프레임: AA 55 | VER 01 | TYPE | SEQ | LEN | PAYLOAD | CRC16/CCITT-FALSE (VER~PAYLOAD, LE)
//   조회(PING/GET_STATUS/GET_CONFIG) → 바로 응답, 동작(HOME/MOVE/STOP/SET_SPEED) → ACK/NAK, HOME·MOVE 는 DONE 으로 끝남
//   응답 100 ms 안 오면 같은 SEQ 로 최대 3회 재전송. DONE 시간 초과 시 이동 명령은 재전송하지 않음
// 연결 경로 두 가지 (내일 현장에서 고름):
//   · USB 직접: 이 PC 의 COM 포트 (Win32 직접 열기, 115200 8N1, DTR·RTS 끔)
//   · 라즈베리파이 경유: 파이의 pi_bridge.py 가 시리얼 바이트를 TCP 로 그대로 넘겨 줌 (같은 프레임)
// 상태: 실제 STM32 와는 아직 시험 안 함 (월요일 현장 첫 연결)
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;

namespace YT
{
    public static class RsProto
    {
        public const byte VER = 1;
        public const byte PING = 0x01, HOME = 0x02, MOVE_REL = 0x03, MOVE_ABS = 0x04, STOP = 0x05, GET_STATUS = 0x06, SET_SPEED = 0x07, GET_CONFIG = 0x08;
        public const byte ACK = 0x81, NAK = 0x82, DONE = 0x83, STATUS = 0x84, PONG = 0x85, CONFIG = 0x86, EVENT = 0x90;
        public static readonly Dictionary<byte, int> LENGTHS = new Dictionary<byte, int> { { ACK, 1 }, { NAK, 2 }, { DONE, 12 }, { STATUS, 20 }, { PONG, 4 }, { CONFIG, 16 }, { EVENT, 9 } };

        public static ushort Crc16(byte[] d, int off, int n)
        {
            int crc = 0xffff;
            for (int i = off; i < off + n; i++)
            {
                crc ^= d[i] << 8;
                for (int b = 0; b < 8; b++) crc = ((crc << 1) ^ ((crc & 0x8000) != 0 ? 0x1021 : 0)) & 0xffff;
            }
            return (ushort)crc;
        }

        public static byte[] Packet(byte type, byte seq, byte[] payload)
        {
            payload = payload ?? new byte[0];
            var f = new byte[8 + payload.Length];
            f[0] = 0xAA; f[1] = 0x55; f[2] = VER; f[3] = type; f[4] = seq; f[5] = (byte)payload.Length;
            Buffer.BlockCopy(payload, 0, f, 6, payload.Length);
            var c = Crc16(f, 2, 4 + payload.Length);
            f[6 + payload.Length] = (byte)(c & 0xff); f[7 + payload.Length] = (byte)(c >> 8);
            return f;
        }

        // 문서 9장 예시 프레임으로 스스로 검사 (PING seq=1 → AA 55 01 01 01 00 75 F6)
        public static bool SelfCheck()
        {
            var p = Packet(PING, 1, null);
            var mv = Packet(MOVE_REL, 12, Pack(-1250, 350));
            return p.Length == 8 && p[6] == 0x75 && p[7] == 0xF6 && mv[14] == 0xA4 && mv[15] == 0x18
                   && Crc16(System.Text.Encoding.ASCII.GetBytes("123456789"), 0, 9) == 0x29B1;
        }

        public static byte[] Pack(int a, int b) { var r = new byte[8]; BitConverter.GetBytes(a).CopyTo(r, 0); BitConverter.GetBytes(b).CopyTo(r, 4); return r; }
        public static byte[] PackU(uint a, uint b) { var r = new byte[8]; BitConverter.GetBytes(a).CopyTo(r, 0); BitConverter.GetBytes(b).CopyTo(r, 4); return r; }
    }

    // Windows COM 포트 (System.IO.Ports 는 Unity .NET Standard 에 없어 Win32 로 직접 엶)
    public static class Win32Com
    {
        [StructLayout(LayoutKind.Sequential)]
        struct DCB
        {
            public int DCBlength; public uint BaudRate; public uint Flags; public ushort wReserved; public ushort XonLim; public ushort XoffLim;
            public byte ByteSize; public byte Parity; public byte StopBits; public sbyte XonChar; public sbyte XoffChar; public sbyte ErrorChar; public sbyte EofChar; public sbyte EvtChar; public ushort wReserved1;
        }
        [StructLayout(LayoutKind.Sequential)]
        struct COMMTIMEOUTS { public uint ReadIntervalTimeout, ReadTotalTimeoutMultiplier, ReadTotalTimeoutConstant, WriteTotalTimeoutMultiplier, WriteTotalTimeoutConstant; }

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr sec, uint disp, uint flags, IntPtr tmpl);
        [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetCommState(SafeFileHandle h, ref DCB dcb);
        [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetCommState(SafeFileHandle h, ref DCB dcb);
        [DllImport("kernel32.dll", SetLastError = true)] static extern bool SetCommTimeouts(SafeFileHandle h, ref COMMTIMEOUTS t);
        [DllImport("kernel32.dll", SetLastError = true)] static extern bool PurgeComm(SafeFileHandle h, uint flags);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern uint QueryDosDevice(string name, char[] buf, int max);

        public static Stream Open(string port)
        {
            var h = CreateFile(@"\\.\" + port, 0xC0000000 /*GENERIC_READ|WRITE*/, 0, IntPtr.Zero, 3 /*OPEN_EXISTING*/, 0, IntPtr.Zero);
            if (h.IsInvalid)
            {
                int e = Marshal.GetLastWin32Error();
                throw new RunError(e == 2 ? $"{port} 포트가 없습니다. STM32 USB 연결을 확인하세요."
                                 : e == 5 ? $"{port} 포트를 다른 프로그램이 쓰고 있습니다 (STM32CubeIDE·터미널 등을 닫으세요)."
                                 : $"{port} 를 열 수 없습니다 (Win32 오류 {e}).");
            }
            var dcb = new DCB { DCBlength = Marshal.SizeOf(typeof(DCB)) };
            GetCommState(h, ref dcb);
            dcb.BaudRate = 115200; dcb.ByteSize = 8; dcb.Parity = 0; dcb.StopBits = 0;   // 8N1
            dcb.Flags = 1;   // fBinary=1, 나머지(패리티 검사·흐름 제어·DTR/RTS) 끔 — serial_stage.py: dtr=False, rts=False
            if (!SetCommState(h, ref dcb)) { h.Dispose(); throw new RunError($"{port} 설정(115200 8N1) 실패"); }
            var t = new COMMTIMEOUTS { ReadIntervalTimeout = uint.MaxValue, ReadTotalTimeoutMultiplier = uint.MaxValue, ReadTotalTimeoutConstant = 20, WriteTotalTimeoutConstant = 1000 };
            SetCommTimeouts(h, ref t);   // 읽기: 바이트가 오면 바로, 없으면 20 ms 후 0 바이트
            PurgeComm(h, 0x0F);
            return new ComStream(h);
        }

        [DllImport("kernel32.dll", SetLastError = true)] static extern bool ReadFile(SafeFileHandle h, byte[] buf, int n, out int read, IntPtr ov);
        [DllImport("kernel32.dll", SetLastError = true)] static extern bool WriteFile(SafeFileHandle h, byte[] buf, int n, out int written, IntPtr ov);

        // ReadFile/WriteFile 를 그대로 쓰는 스트림 (FileStream 이 문자 장치를 다루는 방식에 기대지 않음)
        class ComStream : Stream
        {
            readonly SafeFileHandle h;
            public ComStream(SafeFileHandle h) { this.h = h; }
            public override int Read(byte[] buffer, int offset, int count)
            {
                var tmp = offset == 0 ? buffer : new byte[count];
                if (!ReadFile(h, tmp, count, out int n, IntPtr.Zero)) throw new IOException($"COM 읽기 실패 (Win32 {Marshal.GetLastWin32Error()})");
                if (offset != 0) Buffer.BlockCopy(tmp, 0, buffer, offset, n);
                return n;
            }
            public override void Write(byte[] buffer, int offset, int count)
            {
                var tmp = offset == 0 ? buffer : new byte[count];
                if (offset != 0) Buffer.BlockCopy(buffer, offset, tmp, 0, count);
                if (!WriteFile(h, tmp, count, out int n, IntPtr.Zero) || n != count) throw new IOException($"COM 쓰기 실패 (Win32 {Marshal.GetLastWin32Error()})");
            }
            public override void Flush() { }
            protected override void Dispose(bool disposing) { h.Dispose(); base.Dispose(disposing); }
            public override bool CanRead => true; public override bool CanWrite => true; public override bool CanSeek => false;
            public override long Length => throw new NotSupportedException(); public override long Position { get => 0; set => throw new NotSupportedException(); }
            public override long Seek(long o, SeekOrigin so) => throw new NotSupportedException(); public override void SetLength(long v) => throw new NotSupportedException();
        }

        public static List<string> Ports()
        {
            var r = new List<string>();
            try
            {
                var buf = new char[65536]; uint n = QueryDosDevice(null, buf, buf.Length);
                foreach (var s in new string(buf, 0, (int)n).Split('\0')) if (s.StartsWith("COM")) r.Add(s);
            }
            catch { }
            r.Sort((a, b) => (a.Length == b.Length ? string.CompareOrdinal(a, b) : a.Length - b.Length));
            return r;
        }
    }

    public class LinkStage : IStage
    {
        readonly string name, transport;
        Stream io; TcpClient tcp;
        Thread reader; volatile bool closed;
        readonly object gate = new object();
        readonly object cmdLock = new object();
        byte seq = (byte)new Random().Next(256);
        (byte seq, byte cmd, byte kind, byte[] payload, bool got)? pending;
        (byte seq, byte cmd)? motion;
        volatile string error;
        readonly ConcurrentQueue<DoneInfo> doneQ = new ConcurrentQueue<DoneInfo>();
        readonly ConcurrentQueue<string> logQ = new ConcurrentQueue<string>();
        StStatus lastStatus; float lastStatusAt = -10; StConfig config;
        int[] fw = { 0, 0, 0, 0 };
        int spdY = 100, spdT = 1000;   // 마지막으로 보낸 값 (펌웨어 기본 Y 0.1 mm/s, Θ 1 °/s)

        public event Action<string> Log;
        public event Action<DoneInfo> Done;
        public string Name => name;
        public bool Connected => !closed && error == null;
        public string Mode => "hardware";
        public string Transport => transport;
        public string Error => error;
        public int[] Firmware => fw;
        public StConfig Config => config;
        public int SpeedYUmS => spdY;
        public int SpeedTMdegS => spdT;

        LinkStage(string name, string transport, Stream s, TcpClient t) { this.name = name; this.transport = transport; io = s; tcp = t; }

        public static LinkStage OpenSerial(string port) => Start(new LinkStage($"STM32 · USB 직접 ({port})", "serial", Win32Com.Open(port), null));

        public static LinkStage OpenTcp(string host, int port)
        {
            var c = new TcpClient { NoDelay = true };
            try
            {
                var ar = c.BeginConnect(host, port, null, null);
                if (!ar.AsyncWaitHandle.WaitOne(3000)) { c.Close(); throw new RunError($"파이 {host}:{port} 에 연결할 수 없습니다 (3초 초과). pi_bridge.py 가 켜져 있는지 확인하세요."); }
                c.EndConnect(ar);
            }
            catch (SocketException e) { c.Close(); throw new RunError($"파이 {host}:{port} 연결 실패: {e.Message}"); }
            return Start(new LinkStage($"STM32 · 라즈베리파이 경유 ({host}:{port})", "tcp", c.GetStream(), c));
        }

        // 연결 직후: PING → CONFIG → STATUS 조회만 (장치를 움직이지 않음, demo_web '연결 적용' 과 같음)
        static LinkStage Start(LinkStage s)
        {
            s.reader = new Thread(s.Read) { IsBackground = true, Name = "stm32-rx" }; s.reader.Start();
            try
            {
                var p = s.Exchange(RsProto.PING, null); s.fw = new int[] { p[0], p[1], p[2], p[3] };
                var c = s.Exchange(RsProto.GET_CONFIG, null);
                s.config = new StConfig { yMinUm = BitConverter.ToInt32(c, 0), yMaxUm = BitConverter.ToInt32(c, 4), tMinMdeg = BitConverter.ToInt32(c, 8), tMaxMdeg = BitConverter.ToInt32(c, 12) };
                s.QueryStatus();
            }
            catch { s.Close(); throw; }
            return s;
        }

        void Read()
        {
            var buf = new List<byte>(); var tmp = new byte[256];
            var sw = System.Diagnostics.Stopwatch.StartNew(); long lastByte = 0;
            while (!closed)
            {
                int n;
                try { n = io.Read(tmp, 0, tmp.Length); }
                catch (Exception) { if (!closed) error = tcp != null ? "파이 연결이 끊겼습니다." : "STM32 연결이 끊겼습니다. 장치 상태를 확인하세요."; lock (gate) Monitor.PulseAll(gate); return; }
                if (n == 0)
                {
                    if (tcp != null) { if (!closed) error = "파이 연결이 끊겼습니다."; lock (gate) Monitor.PulseAll(gate); return; }   // TCP: 0 바이트 = 상대가 닫음
                    if (sw.ElapsedMilliseconds - lastByte > 20) buf.Clear();   // USB: 수신 중 20 ms 끊기면 버림 (6.4)
                    continue;
                }
                // TCP 중계는 네트워크 지연으로 프레임 사이가 벌어질 수 있어 20 ms 규칙을 쓰지 않음 (CRC 로 걸러짐)
                lastByte = sw.ElapsedMilliseconds;
                for (int i = 0; i < n; i++) buf.Add(tmp[i]);
                while (buf.Count >= 6)
                {
                    if (buf[0] != 0xAA || buf[1] != 0x55 || buf[2] != RsProto.VER || buf[5] > 32) { buf.RemoveAt(0); continue; }
                    int size = 8 + buf[5];
                    if (buf.Count < size) break;
                    var f = buf.GetRange(0, size).ToArray();
                    if (RsProto.Crc16(f, 2, size - 4) != (ushort)(f[size - 2] | f[size - 1] << 8)) { buf.RemoveAt(0); continue; }
                    buf.RemoveRange(0, size);
                    byte kind = f[3], s = f[4]; var payload = new byte[f[5]]; Buffer.BlockCopy(f, 6, payload, 0, payload.Length);
                    if (RsProto.LENGTHS.TryGetValue(kind, out var len) && len == payload.Length) Receive(kind, s, payload);
                }
            }
        }

        void Receive(byte kind, byte s, byte[] p)
        {
            lock (gate)
            {
                if (kind == RsProto.EVENT)
                {
                    logQ.Enqueue($"← EVENT {(p[0] == 1 ? "LIMIT_HIT" : p[0] == 2 ? "FAULT" : p[0] == 3 ? "BOOT" : p[0].ToString())}");
                    if (p[0] == 3) error = "STM32가 재부팅되었습니다. 다시 연결하고 원점을 설정하세요.";
                }
                else if (kind == RsProto.DONE && motion != null && motion.Value.seq == s && motion.Value.cmd == p[1])
                {
                    var d = new DoneInfo { result = (DoneResult)p[0], cmdType = p[1], yUm = BitConverter.ToInt32(p, 4), tMdeg = BitConverter.ToInt32(p, 8) };
                    motion = null; doneQ.Enqueue(d);
                    if (pending != null && pending.Value.seq == s && pending.Value.cmd == p[1]) pending = (s, p[1], RsProto.ACK, new[] { p[1] }, true);
                }
                else if (pending != null && pending.Value.seq == s)
                {
                    byte cmd = pending.Value.cmd;
                    byte expected = cmd == RsProto.PING ? RsProto.PONG : cmd == RsProto.GET_STATUS ? RsProto.STATUS : cmd == RsProto.GET_CONFIG ? RsProto.CONFIG : RsProto.ACK;
                    if ((kind == RsProto.NAK && p[0] == cmd) || (kind == RsProto.ACK && expected == RsProto.ACK && p[0] == cmd) || (kind == expected && kind != RsProto.ACK && kind != RsProto.NAK))
                        pending = (s, cmd, kind, p, true);
                }
                Monitor.PulseAll(gate);
            }
        }

        // 명령 1개: 100 ms 안에 응답 없으면 같은 SEQ 로 최대 3회 재전송 (6.3)
        byte[] Exchange(byte cmd, byte[] payload) => Exchange(cmd, payload, out _);

        byte[] Exchange(byte cmd, byte[] payload, out NakReason nak)
        {
            nak = NakReason.None;
            lock (cmdLock)
            {
                byte s;
                lock (gate)
                {
                    if (closed || error != null) throw new RunError(error ?? "장치 연결이 닫혔습니다.");
                    seq = (byte)(seq + 1); s = seq;
                    pending = (s, cmd, (byte)0, (byte[])null, false);
                    if (cmd == RsProto.HOME || cmd == RsProto.MOVE_REL || cmd == RsProto.MOVE_ABS) motion = (s, cmd);
                }
                var frame = RsProto.Packet(cmd, s, payload);
                try
                {
                    for (int attempt = 0; attempt < 4; attempt++)
                    {
                        lock (gate) if (pending.Value.got) break;
                        io.Write(frame, 0, frame.Length); io.Flush();
                        lock (gate)
                        {
                            var until = DateTime.UtcNow.AddMilliseconds(100);
                            while (!pending.Value.got && error == null && !closed)
                            {
                                var left = until - DateTime.UtcNow; if (left <= TimeSpan.Zero) break;
                                Monitor.Wait(gate, left);
                            }
                            if (pending.Value.got) break;
                        }
                    }
                    lock (gate)
                    {
                        if (error != null || closed) throw new RunError(error ?? "장치 연결이 닫혔습니다.");
                        var r = pending.Value;
                        if (!r.got) { if (motion != null && motion.Value.seq == s) motion = null; throw new RunError("STM32 응답 시간 초과. 실제 이동 여부를 확인하세요."); }
                        if (r.kind == RsProto.NAK) { nak = (NakReason)r.payload[1]; if (motion != null && motion.Value.seq == s) motion = null; return null; }
                        return r.payload;
                    }
                }
                catch (IOException e) { throw new RunError("STM32 전송 실패: " + e.Message); }
                finally { lock (gate) pending = null; }
            }
        }

        void QueryStatus()
        {
            var p = Exchange(RsProto.GET_STATUS, null);
            lastStatus = new StStatus
            {
                state = (StState)p[0], flags = (StFlags)p[1], yUm = BitConverter.ToInt32(p, 4), tMdeg = BitConverter.ToInt32(p, 8),
                ySteps = BitConverter.ToInt32(p, 12), tSteps = BitConverter.ToInt32(p, 16),
            };
            lastStatusAt = UnityEngine.Time.realtimeSinceStartup;
        }

        // 너무 자주 묻지 않게 0.1 초 안의 재질의는 직전 값 사용
        public StStatus Status
        {
            get
            {
                if (UnityEngine.Time.realtimeSinceStartup - lastStatusAt > 0.1f && error == null) QueryStatus();
                return lastStatus;
            }
        }

        public int[] Ping() { var p = Exchange(RsProto.PING, null); fw = new int[] { p[0], p[1], p[2], p[3] }; return fw; }

        NakReason Act(byte cmd, byte[] payload, string text)
        {
            Emit("→ " + text);
            Exchange(cmd, payload, out var nak);
            Emit(nak == NakReason.None ? "← ACK" : $"← NAK ({StText.Nak(nak)})");
            if (nak == NakReason.None && cmd != RsProto.STOP && cmd != RsProto.SET_SPEED) lastStatusAt = -10;
            return nak;
        }

        public NakReason Home(int mask) => Act(RsProto.HOME, new[] { (byte)mask }, $"HOME mask={mask}");
        public NakReason MoveAbs(int y, int t) => Act(RsProto.MOVE_ABS, RsProto.Pack(y, t), $"MOVE_ABS Y {y} µm, Θ {t} mdeg");
        public NakReason MoveRel(int y, int t) => Act(RsProto.MOVE_REL, RsProto.Pack(y, t), $"MOVE_REL Y {y} µm, Θ {t} mdeg");
        public NakReason Stop() => Act(RsProto.STOP, null, "STOP");
        public NakReason SetSpeed(int y, int t)
        {
            var r = Act(RsProto.SET_SPEED, RsProto.PackU((uint)Math.Max(0, y), (uint)Math.Max(0, t)), $"SET_SPEED Y {y} µm/s, Θ {t} mdeg/s");
            if (r == NakReason.None) { if (y > 0) spdY = y; if (t > 0) spdT = t; }
            return r;
        }

        void Emit(string s) => logQ.Enqueue(s);

        // 메인 스레드: DONE·기록을 화면 쪽 이벤트로 넘김
        public void Tick(float dt)
        {
            while (logQ.TryDequeue(out var s)) Log?.Invoke(s);
            while (doneQ.TryDequeue(out var d))
            {
                lastStatusAt = -10;
                Log?.Invoke($"← DONE {StText.Done(d.result)} · Y {StText.Mm(d.yUm)} mm / Θ {StText.Mm(d.tMdeg)}°");
                Done?.Invoke(d);
            }
        }

        public void Close()
        {
            closed = true;
            lock (gate) Monitor.PulseAll(gate);
            try { io?.Dispose(); } catch { }
            try { tcp?.Close(); } catch { }
        }
    }
}
