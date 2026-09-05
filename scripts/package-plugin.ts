import { createHash } from 'node:crypto';
import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

const repositoryRoot = resolve(import.meta.dir, '..');
const pluginRoot = join(repositoryRoot, 'plugins', 'sonicboom.tts');
const defaultOutput = join(repositoryRoot, 'dist', 'plugins');
const stagingRoot = join(defaultOutput, '.staging');
const pluginId = 'sonicboom.tts';
const binaryName = 'sonicboom-tiktools-plugin';

function fail(message: string): never {
  throw new Error(`TikTools plugin packaging failed: ${message}`);
}

function run(command: string, args: string[]): void {
  const result = Bun.spawnSync({
    cmd: [command, ...args],
    cwd: repositoryRoot,
    stdout: 'inherit',
    stderr: 'inherit',
  });
  if (!result.success) fail(`${command} ${args.join(' ')} exited with code ${result.exitCode}`);
}

function capture(command: string, args: string[]): string {
  const result = Bun.spawnSync({
    cmd: [command, ...args],
    cwd: repositoryRoot,
    stdout: 'pipe',
    stderr: 'inherit',
  });
  if (!result.success) fail(`${command} ${args.join(' ')} exited with code ${result.exitCode}`);
  return result.stdout ? new TextDecoder().decode(result.stdout) : '';
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function collectFiles(directory: string, base: string): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const fullPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectFiles(fullPath, base)));
    } else if (entry.isFile()) {
      files.push(fullPath.slice(base.length + 1).replaceAll('\\', '/'));
    }
  }
  return files;
}

const args = process.argv.slice(2);
const debug = args.includes('--debug');
const outputArgument = args.find((arg) => arg.startsWith('--out='));
if (args.some((arg) => !['--debug', outputArgument].includes(arg))) {
  fail('usage: bun run scripts/package-plugin.ts [--debug] [--out=<directory>]');
}

const outputDirectory = resolve(
  repositoryRoot,
  outputArgument?.slice('--out='.length) || process.env.PLUGIN_OUT_DIR?.trim() || defaultOutput,
);
const profile = debug ? 'debug' : 'release';
const binaryFile = process.platform === 'win32' ? `${binaryName}.exe` : binaryName;
const builtBinary = join(repositoryRoot, 'target', profile, binaryFile);

if (!(await exists(join(pluginRoot, 'plugin.json')))) {
  fail(`missing manifest under ${pluginRoot}`);
}

run('cargo', [
  'build',
  ...(debug ? [] : ['--release']),
  '--no-default-features',
  '--features',
  'tiktools-plugin',
  '--bin',
  binaryName,
  '--locked',
]);
if (!(await exists(builtBinary))) fail(`built binary not found at ${builtBinary}`);

const manifest = JSON.parse(await readFile(join(pluginRoot, 'plugin.json'), 'utf8')) as Record<
  string,
  unknown
>;
const stagedEntry = binaryFile;
const packageDirectory = join(stagingRoot, pluginId);
const archivePath = join(outputDirectory, `${pluginId}.plugin`);
const temporaryArchive = join(stagingRoot, `${pluginId}.zip`);

await mkdir(outputDirectory, { recursive: true });
await mkdir(stagingRoot, { recursive: true });
await rm(packageDirectory, { recursive: true, force: true });
await rm(temporaryArchive, { force: true });
await rm(archivePath, { force: true });
await mkdir(dirname(join(packageDirectory, stagedEntry)), { recursive: true });
await cp(builtBinary, join(packageDirectory, stagedEntry));
await writeFile(
  join(packageDirectory, 'plugin.json'),
  `${JSON.stringify({ ...manifest, entry: stagedEntry }, null, 2)}\n`,
  'utf8',
);

const checksums: Record<string, string> = {};
for (const relative of (await collectFiles(packageDirectory, packageDirectory)).sort()) {
  if (relative === 'checksums.json' || relative === 'signature.json') continue;
  checksums[relative] = createHash('sha256')
    .update(await readFile(join(packageDirectory, relative)))
    .digest('hex');
}
await writeFile(join(packageDirectory, 'checksums.json'), `${JSON.stringify(checksums, null, 2)}\n`);

run('tar', ['-a', '-c', '-f', temporaryArchive, '-C', stagingRoot, pluginId]);
await cp(temporaryArchive, archivePath);
await rm(temporaryArchive, { force: true });

const listing = capture('tar', ['-tf', archivePath])
  .split(/\r?\n/)
  .map((entry) => entry.replaceAll('\\', '/'));
for (const expected of [`${pluginId}/plugin.json`, `${pluginId}/checksums.json`]) {
  if (!listing.includes(expected)) fail(`archive is missing ${expected}`);
}

console.log(`Created ${basename(archivePath)}`);
