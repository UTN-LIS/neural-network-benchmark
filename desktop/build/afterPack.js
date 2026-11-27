'use strict';

const fs = require('node:fs');
const path = require('node:path');

/**
 * Ensure the Python sidecar stays executable after electron-builder copies it
 * into the app bundle. Windows does not need this; macOS/Linux do.
 */
module.exports = async function afterPack(context) {
  let executable = null;

  if (context.electronPlatformName === 'darwin') {
    executable = path.join(
      context.appOutDir,
      `${context.packager.appInfo.productFilename}.app`,
      'Contents',
      'Resources',
      'backend',
      'prebenchmark-sidecar'
    );
  } else if (context.electronPlatformName === 'linux') {
    executable = path.join(
      context.appOutDir,
      'resources',
      'backend',
      'prebenchmark-sidecar'
    );
  }

  if (executable && fs.existsSync(executable)) {
    fs.chmodSync(executable, 0o755);
  }
};
