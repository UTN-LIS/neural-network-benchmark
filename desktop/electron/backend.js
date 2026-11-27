'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');

const HEALTH_TIMEOUT_MS = 120_000;
const HEALTH_POLL_MS = 300;
const GRACEFUL_STOP_MS = 8_000;
const FORCE_STOP_MS = 3_000;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

class SidecarBackend {
  constructor({ dataDir, logFile, isPackaged }) {
    this.dataDir = dataDir;
    this.logFile = logFile;
    this.isPackaged = isPackaged;
    this.child = null;
    this.port = null;
    this.exited = false;
    this.logStream = null;
  }

  get executableName() {
    return process.platform === 'win32' ? 'prebenchmark-sidecar.exe' : 'prebenchmark-sidecar';
  }

  resolveExecutable() {
    const candidates = this.isPackaged
      ? [path.join(process.resourcesPath, 'backend', this.executableName)]
      : [path.join(__dirname, '..', 'backend', 'dist', 'prebenchmark-sidecar', this.executableName)];

    const found = candidates.find((candidate) => fs.existsSync(candidate));
    if (!found) {
      throw new Error(
        `No se encontró el backend empaquetado (${this.executableName}). ` +
          'Ejecutá "npm run build:backend" antes de iniciar la app.'
      );
    }
    return found;
  }

  log(message) {
    const line = `[${new Date().toISOString()}] ${message}\n`;
    if (this.logStream) this.logStream.write(line);
    if (!this.isPackaged) process.stdout.write(line);
  }

  async start() {
    this.port = await findFreePort();
    const executable = this.resolveExecutable();

    fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
    this.logStream = fs.createWriteStream(this.logFile, { flags: 'a' });
    this.log(`starting ${executable} on 127.0.0.1:${this.port}`);

    this.child = spawn(
      executable,
      ['--data-dir', this.dataDir, '--port', String(this.port)],
      {
        cwd: this.dataDir,
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONUTF8: '1',
          PYTHONIOENCODING: 'utf-8',
          PREBENCHMARK_TORCH_THREADS: process.env.PREBENCHMARK_TORCH_THREADS || '0',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    );

    this.child.stdout.setEncoding('utf8');
    this.child.stderr.setEncoding('utf8');
    this.child.stdout.on('data', (chunk) => this.log(`stdout: ${chunk.trimEnd()}`));
    this.child.stderr.on('data', (chunk) => this.log(`stderr: ${chunk.trimEnd()}`));
    this.child.stdin.on('error', () => {
      // The process may exit before we write the shutdown line.
    });
    this.child.on('error', (error) => {
      this.spawnError = error;
      this.exited = true;
      this.log(`spawn error: ${error.message}`);
    });
    this.child.once('exit', (code, signal) => {
      this.exited = true;
      this.log(`backend exited code=${code} signal=${signal}`);
    });

    await this.waitForHealth();
    return `http://127.0.0.1:${this.port}`;
  }

  async waitForHealth() {
    const deadline = Date.now() + HEALTH_TIMEOUT_MS;
    const url = `http://127.0.0.1:${this.port}/api/health`;

    while (Date.now() < deadline) {
      if (this.exited) {
        const detail = this.spawnError ? `: ${this.spawnError.message}` : '';
        throw new Error(`El backend no pudo iniciar${detail}. Revisá el log.`);
      }
      try {
        const response = await fetch(url, { signal: AbortSignal.timeout(2_000) });
        if (response.ok) {
          const body = await response.json();
          if (body && body.status === 'ok') return body;
        }
      } catch {
        // Not ready yet.
      }
      await delay(HEALTH_POLL_MS);
    }

    throw new Error(
      `El backend no respondió en ${HEALTH_TIMEOUT_MS / 1000} segundos. Revisá el log.`
    );
  }

  waitForExit(timeoutMs) {
    return new Promise((resolve) => {
      if (this.exited || !this.child) {
        resolve(true);
        return;
      }
      const onExit = () => {
        clearTimeout(timer);
        resolve(true);
      };
      const timer = setTimeout(() => {
        this.child.off('exit', onExit);
        resolve(false);
      }, timeoutMs);
      this.child.once('exit', onExit);
    });
  }

  async stop() {
    if (!this.child || this.exited) {
      if (this.logStream) this.logStream.end();
      return;
    }

    this.log('requesting graceful shutdown');
    try {
      this.child.stdin.write('shutdown\n');
      this.child.stdin.end();
    } catch {
      // Fall through to the forced kill.
    }

    const graceful = await this.waitForExit(GRACEFUL_STOP_MS);
    if (!graceful) {
      this.log('forcing backend kill');
      this.child.kill();
      await this.waitForExit(FORCE_STOP_MS);
    }
    if (this.logStream) this.logStream.end();
  }
}

module.exports = { SidecarBackend, findFreePort };
