'use strict';

const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const fs = require('node:fs');
const path = require('node:path');

const { SidecarBackend } = require('./backend');

const APP_NAME = 'Neural Network Benchmark';
const APP_ID = 'com.utnlis.neural-network-benchmark';

app.setName(APP_NAME);
if (process.platform === 'win32') {
  app.setAppUserModelId(APP_ID);
}

let mainWindow = null;
let splashWindow = null;
let backend = null;
let quitting = false;
let quittingReady = false;

/**
 * Large, machine-local data must not land in a roaming profile on Windows
 * domain machines. Elsewhere Electron's per-user directory is the right place.
 */
function appDataRoot() {
  if (process.platform === 'win32' && process.env.LOCALAPPDATA) {
    return path.join(process.env.LOCALAPPDATA, APP_NAME);
  }
  return app.getPath('userData');
}

function dataDir() {
  return path.join(appDataRoot(), 'benchmark');
}

function logDir() {
  return path.join(appDataRoot(), 'logs');
}

function logLine(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  try {
    fs.appendFileSync(path.join(logDir(), 'desktop.log'), line);
  } catch {
    // Logging must never prevent startup.
  }
  if (!app.isPackaged) process.stdout.write(line);
}

function createSplash() {
  splashWindow = new BrowserWindow({
    width: 560,
    height: 340,
    frame: false,
    resizable: false,
    show: false,
    backgroundColor: '#0b1220',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  splashWindow.once('ready-to-show', () => splashWindow.show());
  splashWindow.on('closed', () => {
    splashWindow = null;
  });
}

function closeSplash() {
  if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
  splashWindow = null;
}

function buildMenu() {
  if (process.platform !== 'darwin') {
    Menu.setApplicationMenu(null);
    return;
  }
  const template = [
    {
      label: APP_NAME,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { label: 'Editar', submenu: [{ role: 'undo' }, { role: 'redo' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'Ver', submenu: app.isPackaged ? [{ role: 'togglefullscreen' }] : [{ role: 'reload' }, { role: 'toggleDevTools' }, { type: 'separator' }, { role: 'togglefullscreen' }] },
    { label: 'Ventana', submenu: [{ role: 'minimize' }, { role: 'zoom' }, { role: 'front' }] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createMainWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1024,
    minHeight: 720,
    show: false,
    title: APP_NAME,
    backgroundColor: '#f7f8fa',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    if (/^https?:/i.test(target)) shell.openExternal(target);
    return { action: 'deny' };
  });

  mainWindow.webContents.on('will-navigate', (event, target) => {
    if (!target.startsWith(url)) event.preventDefault();
  });

  mainWindow.webContents.on('did-fail-load', (_event, code, description, failedUrl, isMainFrame) => {
    if (!isMainFrame || code === -3) return;
    logLine(`did-fail-load ${code} ${description} ${failedUrl}`);
    dialog.showErrorBox(
      'No se pudo cargar la interfaz',
      `${description} (${code})\n${failedUrl}`
    );
  });

  mainWindow.once('ready-to-show', () => {
    closeSplash();
    mainWindow.show();
  });

  // Keep the OS window title stable; the web document title is page-specific.
  mainWindow.on('page-title-updated', (event) => event.preventDefault());

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.loadURL(url);
}

async function bootstrap() {
  fs.mkdirSync(dataDir(), { recursive: true });
  fs.mkdirSync(logDir(), { recursive: true });

  createSplash();
  buildMenu();

  backend = new SidecarBackend({
    dataDir: dataDir(),
    logFile: path.join(logDir(), 'backend.log'),
    isPackaged: app.isPackaged,
  });

  try {
    const url = await backend.start();
    logLine(`backend ready at ${url}`);
    createMainWindow(url);
  } catch (error) {
    logLine(`startup failed: ${error && error.stack ? error.stack : error}`);
    closeSplash();
    dialog.showErrorBox(
      'No se pudo iniciar el entorno',
      `${error.message}\n\nLog: ${path.join(logDir(), 'backend.log')}`
    );
    quittingReady = true;
    app.quit();
  }
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(bootstrap);
  app.on('window-all-closed', () => app.quit());

  app.on('before-quit', (event) => {
    if (quittingReady) return;
    event.preventDefault();
    if (quitting) return;
    quitting = true;
    const finish = () => {
      quittingReady = true;
      app.quit();
    };
    if (backend) backend.stop().finally(finish);
    else finish();
  });
}
