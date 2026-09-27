const { spawn } = require('node:child_process');
const http = require('node:http');

const apiUrl = 'http://127.0.0.1:8001/api/locations';
let api;
let vite;

function isApiReady() {
  return new Promise((resolve) => {
    const request = http.get(apiUrl, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.setTimeout(500, () => { request.destroy(); resolve(false); });
    request.on('error', () => resolve(false));
  });
}

async function waitForApi() {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    if (await isApiReady()) return true;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

function stop() {
  if (vite && !vite.killed) vite.kill();
  if (api && !api.killed) api.kill();
}

async function main() {
  if (!(await isApiReady())) {
    api = spawn('python', ['server.py'], { cwd: __dirname + '/..', stdio: 'inherit', env: { ...process.env, PORT: '8001' } });
    if (!(await waitForApi())) {
      stop();
      process.exitCode = 1;
      return;
    }
  }
  vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js'], { cwd: __dirname + '/..', stdio: 'inherit' });
  vite.on('exit', (code) => { stop(); process.exitCode = code ?? 0; });
}

process.on('SIGINT', stop);
process.on('SIGTERM', stop);
main();
