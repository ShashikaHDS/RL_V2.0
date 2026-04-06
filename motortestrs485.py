#!/usr/bin/env python3
import time, argparse, sys
import minimalmodbus, serial

# Registers (TinS-6 style)
REG_MODE = 0x50    # 1 = speed closed-loop
REG_RUN  = 0x51    # 1 = start, 0 = stop
REG_LSPD = 0x0C    # left  speed target (lines/s, signed)
REG_RSPD = 0x0D    # right speed target (lines/s, signed)

def make_ins(port, sid, baud, parity, stopbits, timeout, rs485_toggle, debug):
    ins = minimalmodbus.Instrument(port, sid, mode=minimalmodbus.MODE_RTU)
    ins.serial.baudrate = baud
    ins.serial.bytesize = 8
    ins.serial.parity   = parity         # <-- try EVEN first (8E1)
    ins.serial.stopbits = stopbits       # <-- 1
    ins.serial.timeout  = timeout        # generous
    ins.clear_buffers_before_each_transaction = True
    ins.handle_local_echo = True         # harmless; helps with echo-y dongles
    if rs485_toggle:
        try:
            from serial.rs485 import RS485Settings
            ins.serial.rs485_mode = RS485Settings(
                rts_level_for_tx=True, rts_level_for_rx=False,
                delay_before_tx=0, delay_before_rx=0
            )
        except Exception:
            pass
    ins.debug = debug
    return ins

def try_read(ins, reg=0x50):
    try:
        val = ins.read_register(reg, 0, functioncode=3, signed=False)
        print(f"[READ OK] reg 0x{reg:X} = {val}")
        return True
    except Exception as e:
        msg = str(e)
        # Any Modbus *exception* string means framing is correct and device replied
        if any(k in msg for k in ["Illegal", "Device", "Slave", "Acknowledge", "Busy"]):
            print(f"[ALIVE] Modbus exception (framing OK): {msg}")
            return True
        print(f"[READ FAIL] {msg}")
        return False

def w_u16(ins, reg, val):
    # Prefer FC16; fall back to FC6 if the slave expects it
    try:
        ins.write_register(reg, int(val) & 0xFFFF, 0, functioncode=16, signed=False)
    except Exception:
        ins.write_register(reg, int(val) & 0xFFFF, 0, functioncode=6, signed=False)

def w_s16(ins, reg, val):
    try:
        ins.write_register(reg, int(val) & 0xFFFF, 0, functioncode=16, signed=True)
    except Exception:
        ins.write_register(reg, int(val) & 0xFFFF, 0, functioncode=6, signed=True)

def feed(ins, lpsL, lpsR, dur, period=0.05):
    t0 = time.time()
    while time.time() - t0 < dur:
        w_s16(ins, REG_LSPD, lpsL)
        w_s16(ins, REG_RSPD, lpsR)
        time.sleep(period)

def main():
    ap = argparse.ArgumentParser(description="RS-485 motor test with 8E1 framing + FC16 writes.")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--baud", type=int, default=19200)      # try 19200 first
    ap.add_argument("--alt-baud", type=int, default=38400)  # fallback baud to try next
    ap.add_argument("--parity", choices=["even","none"], default="even")
    ap.add_argument("--stopbits", type=int, choices=[1,2], default=1)
    ap.add_argument("--timeout", type=float, default=0.6)
    ap.add_argument("--rs485-toggle", action="store_true", help="Enable RTS for MAX485 adapters")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--lines", type=int, default=1200)
    ap.add_argument("--fwd", type=float, default=3.0)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--bwd", type=float, default=3.0)
    args = ap.parse_args()

    parity = serial.PARITY_EVEN if args.parity == "even" else serial.PARITY_NONE

    # First attempt: 8E1 @ --baud
    ins = make_ins(args.port, args.slave, args.baud, parity, args.stopbits, args.timeout, args.rs485_toggle, args.debug)
    print(f"[TRY] baud={args.baud}, parity={'EVEN' if parity==serial.PARITY_EVEN else 'NONE'}, stop={args.stopbits}, id={args.slave}")
    if not try_read(ins, 0x50):
        # Fallback: try alt baud with same parity/stopbits
        print(f"[RETRY] Trying alt baud {args.alt_baud} with same framing…")
        ins = make_ins(args.port, args.slave, args.alt_baud, parity, args.stopbits, args.timeout, args.rs485_toggle, args.debug)
        if not try_read(ins, 0x50):
            print("[FAIL] Still not alive. This is almost always wrong wires/framing:")
            print("       Use A (D+), B (D−), and GND; swap A/B; try the other twisted pair (CAN vs RS-485).")
            sys.exit(1)

    # If we got here, framing is OK. Do motion.
    print("[CMD] MODE=1 (speed-closed-loop)"); w_u16(ins, REG_MODE, 1)
    print("[CMD] RUN=1");                      w_u16(ins, REG_RUN, 1)

    try:
        print(f"[MOVE] Forward {args.lines} lps for {args.fwd}s")
        feed(ins, +args.lines, +args.lines, args.fwd)

        print(f"[MOVE] Stop {args.pause}s")
        feed(ins, 0, 0, args.pause)

        print(f"[MOVE] Backward {-args.lines} lps for {args.bwd}s")
        feed(ins, -args.lines, -args.lines, args.bwd)

        feed(ins, 0, 0, 0.5)
        print("[DONE]")
    finally:
        try:
            print("[CMD] RUN=0"); w_u16(ins, REG_RUN, 0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
