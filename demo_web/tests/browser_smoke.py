"""Exercise the actual UI with a temporary database and the system Chromium.

Run from repo root: demo_web/.venv/bin/python demo_web/tests/browser_smoke.py
No physical device is opened and no real admin credential is changed.
"""
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright, expect
from werkzeug.security import generate_password_hash
from demo_web.store import Store


def main():
    artifacts = ROOT / 'demo_web' / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='alignment-browser-') as directory:
        password = secrets.token_urlsafe(20)
        store = Store(Path(directory) / 'demo.sqlite3')
        store.meta('password', generate_password_hash(password))
        store.db.close()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        process = subprocess.Popen([sys.executable, '-m', 'demo_web', 'serve', '--data-dir', directory,
                                    '--port', str(port)], cwd=ROOT, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(base+'/api/state', timeout=.3).close()
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise RuntimeError('QA server failed to start')
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True)
                context = browser.new_context(viewport={'width':1440, 'height':900})
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base+'/login')
                page.locator('#password').fill(password)
                page.get_by_role('button', name='로그인 →').click()
                page.wait_for_url(base+'/admin')
                expect(page.locator('#device-home')).to_have_text('설정됨', timeout=10000)
                # Never click Y HOME here: it always drives the real STM32, even in mock mode.
                expect(page.locator('#home-y')).to_be_enabled()
                page.locator('#jog-plus').click()
                expect(page.locator('#device-y')).to_have_text('0.100', timeout=10000)
                expect(page.locator('#test-trial')).to_have_text('0 / 20')
                page.locator('#jog-minus').click()
                expect(page.locator('#device-y')).to_have_text('0.000', timeout=10000)
                expect(page.locator('#status-title')).to_have_text('Y축 수동 시험 완료', timeout=10000)
                page.locator('#camera-profile').select_option('preview')
                page.locator('#camera-apply').click()
                expect(page.locator('#camera-settings-status')).to_contain_text('960 × 720')
                page.locator('#camera-profile').select_option('full')
                page.locator('#camera-apply').click()
                expect(page.locator('#camera-settings-status')).to_contain_text('4056 × 3040')
                expect(page.locator('#camera-undistort')).to_have_value('true')
                page.locator('#camera-undistort').select_option('false')
                page.locator('#camera-apply').click()
                expect(page.locator('#camera-settings-status')).to_contain_text('왜곡 보정 꺼짐')
                page.reload()
                expect(page.locator('#camera-undistort')).to_have_value('false')
                page.locator('#camera-undistort').select_option('true')
                page.locator('#camera-apply').click()
                expect(page.locator('#camera-settings-status')).to_contain_text('왜곡 보정 켜짐')
                page.locator('#example').click()
                page.locator('#save-scenario').click()
                expect(page.locator('#editor-heading')).to_contain_text('v1')
                page.locator('#test-prepare').click()
                expect(page.locator('#status-title')).to_have_text('보정 대기', timeout=10000)
                expect(page.locator('#test-trial')).to_have_text('0 / 20')
                expect(page.locator('#test-y')).to_have_text('0.000')
                page.locator('#test-step').click()
                expect(page.locator('#test-trial')).to_have_text('1 / 20')
                expect(page.locator('#status-title')).to_have_text('보정 대기', timeout=10000)
                page.locator('#test-auto').click()
                expect(page.locator('#status-title')).to_have_text('보정 완료', timeout=15000)
                expect(page.locator('#test-trial')).to_have_text('5 / 20')
                page.locator('#verify-confirm').check()
                page.locator('#verify').click()
                expect(page.locator('#verification-badge')).to_have_text('모의 검증')
                page.locator('#activate').click()
                expect(page.locator('#activate')).to_have_text('시연 비활성화')
                page.locator('#refresh-history').click()
                page.evaluate('scrollTo(0, 0)')
                page.screenshot(path=str(artifacts/'admin-desktop.png'), full_page=True)
                page.goto(base+'/')
                page.locator('#random').click()
                expect(page.locator('#status-title')).to_have_text('보정 대기', timeout=10000)
                expect(page.locator('#trial')).to_have_text('00')
                expect(page.locator('#y-value')).to_have_text('0.000')
                page.locator('#correct').click()
                expect(page.locator('#status-title')).to_have_text('보정 완료', timeout=15000)
                expect(page.locator('#trial')).to_have_text('05')
                expect(page.locator('#y-value')).to_have_text('-0.200')
                assert page.evaluate('document.documentElement.scrollHeight <= innerHeight'), 'Presentation overflows vertically'
                page.screenshot(path=str(artifacts/'demo-desktop.png'), full_page=True)
                # Reload restores server state without issuing a new move.
                page.reload()
                expect(page.locator('#trial')).to_have_text('05')
                expect(page.locator('#status-title')).to_have_text('보정 완료')
                state = page.request.get(base+'/api/state').json()
                assert 'snapshot' not in state and 'position' not in state
                assert state['y_um'] == -200
                page.set_viewport_size({'width':390, 'height':844})
                page.screenshot(path=str(artifacts/'demo-mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), page.evaluate('[...document.querySelectorAll("body *")].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,cls:e.className,right:e.getBoundingClientRect().right})).slice(0,20)')
                page.goto(base+'/admin')
                expect(page.locator('#editor-heading')).to_contain_text('v1')
                page.screenshot(path=str(artifacts/'admin-mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), page.evaluate('[...document.querySelectorAll("body *")].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,cls:e.className,right:e.getBoundingClientRect().right})).slice(0,20)')
                page.locator('#logout').click()
                page.wait_for_url(base+'/')
                expect(page.locator('#random')).to_be_disabled()
                assert not errors, errors
                browser.close()
                print('Browser PASS: login → create → prepare → single step → automatic correction → verify → activate → demo → reload → mobile → logout')
                print(f'Screenshots: {artifacts}')
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
