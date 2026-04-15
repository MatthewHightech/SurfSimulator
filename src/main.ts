import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PhysicsEngine } from './Physics/PhysicsEngine';
import { SSFManager } from './Graphics/SSFManager';
import { PISTON, REEF, tankVisual, PHYSICS_UNIFORM_DEFAULTS } from './sim/tankSceneConfig';
import './styles/app.css';

/** Default raised after merging density+forces into one compute pass (~2× less neighbor work). */
const PARTICLE_COUNT = 200_000;

let physics: PhysicsEngine;
let ssf: SSFManager;

const viewportRoot = document.querySelector('#viewport');
if (!(viewportRoot instanceof HTMLDivElement)) {
  throw new Error('Missing #viewport container');
}
const viewportEl = viewportRoot;

const renderer = new WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

function resizeRendererToViewport() {
  const w = Math.max(1, viewportEl.clientWidth);
  const h = Math.max(1, viewportEl.clientHeight);
  renderer.setSize(w, h, false);
}

resizeRendererToViewport();
viewportEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);

const INITIAL_CAMERA_POSITION = new THREE.Vector3(20, 20, 20);
const INITIAL_CONTROLS_TARGET = new THREE.Vector3(0, 0, 0);

const camera = new THREE.PerspectiveCamera(
  75,
  viewportEl.clientWidth / Math.max(1, viewportEl.clientHeight),
  0.1,
  1000,
);
camera.position.copy(INITIAL_CAMERA_POSITION);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const viewportObserver = new ResizeObserver(() => {
  resizeRendererToViewport();
  camera.aspect = viewportEl.clientWidth / Math.max(1, viewportEl.clientHeight);
  camera.updateProjectionMatrix();
});
viewportObserver.observe(viewportEl);

// Helpers
scene.add(new THREE.GridHelper(tankVisual.width, tankVisual.width, 0x222222, 0x111111));
scene.add(new THREE.AxesHelper(5));

const tankGeo = new THREE.BoxGeometry(tankVisual.width, tankVisual.height, tankVisual.depth);
const tankMesh = new THREE.Mesh(
  tankGeo,
  new THREE.MeshStandardMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.1,
    side: THREE.BackSide,
  }),
);
tankMesh.position.y = tankVisual.centerY;
scene.add(tankMesh);

const pistonMesh = new THREE.Mesh(
  new THREE.BoxGeometry(PISTON.thickness, tankVisual.height, tankVisual.depth),
  new THREE.MeshStandardMaterial({ color: 0x444444 }),
);
pistonMesh.position.x = PISTON.restX;
scene.add(pistonMesh);

const reefMesh = new THREE.Mesh(
  new THREE.BoxGeometry(REEF.halfExtents.x * 2, REEF.halfExtents.y * 2, REEF.halfExtents.z * 2),
  new THREE.MeshStandardMaterial({ color: 0x222222 })
);
reefMesh.position.set(REEF.center.x, REEF.center.y, REEF.center.z);
scene.add(reefMesh);


// Ambient Light'
scene.add(new THREE.AmbientLight(0xffffff, 1));

// add directional light
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(10, 10, 10);
scene.add(directionalLight);
const clock = new THREE.Timer();

/** Elapsed simulation time (s), drives shader recycle / swell — resettable. */
let simulationTime = 0;

/** One-shot piston surge after “Send wave” (`performance.now()` ms). */
let surgeStartMs: number | null = null;

function getPistonXFromSurge(): number {
  const rest = PISTON.restX;
  if (surgeStartMs === null) return rest;
  const elapsed = performance.now() - surgeStartMs;
  if (elapsed >= PISTON.surgeDurationMs) {
    surgeStartMs = null;
    return rest;
  }
  const t = elapsed / PISTON.surgeDurationMs;
  const factor = Math.sin(Math.PI * t);
  return rest + PISTON.surgeAmplitude * factor;
}

function beginSurge(): void {
  surgeStartMs = performance.now();
}

function cancelSurge(): void {
  surgeStartMs = null;
}

function applyDefaultPhysicsUniforms(engine: PhysicsEngine): void {
  engine.deltaTime.value = PHYSICS_UNIFORM_DEFAULTS.deltaTime;
  engine.swellAmplitude.value = PHYSICS_UNIFORM_DEFAULTS.swellAmplitude;
  engine.swellPeriod.value = PHYSICS_UNIFORM_DEFAULTS.swellPeriod;
  engine.swellSpeed.value = PHYSICS_UNIFORM_DEFAULTS.swellSpeed;
}

function resetSimulationToInitialState(engine: PhysicsEngine): void {
  cancelSurge();
  simulationTime = 0;
  pistonMesh.position.x = PISTON.restX;
  engine.pistonX.value = PISTON.restX;
  engine.uTime.value = 0;
  applyDefaultPhysicsUniforms(engine);
  engine.resetParticlesToInitialState();
  camera.position.copy(INITIAL_CAMERA_POSITION);
  controls.target.copy(INITIAL_CONTROLS_TARGET);
  controls.update();
}

async function init() {
  try {
    await renderer.init();
    console.log("WebGPU Initialized.");

    physics = new PhysicsEngine(PARTICLE_COUNT);
    
    ssf = new SSFManager(renderer, physics);

    const sendWaveBtn = document.querySelector('#btn-send-wave');
    const stillOceanBtn = document.querySelector('#btn-still-ocean');
    const resetBtn = document.querySelector('#btn-reset');
    sendWaveBtn?.addEventListener('click', () => beginSurge());
    stillOceanBtn?.addEventListener('click', () => cancelSurge());
    resetBtn?.addEventListener('click', () => resetSimulationToInitialState(physics));

    animate();
  } catch (e) {
    console.error("WebGPU error:", e);
  }
}

function animate() {
  requestAnimationFrame(animate);
  clock.update();
  
  const delta = clock.getDelta();
  simulationTime += Math.min(delta, 0.033);

  if (physics && ssf) {
    physics.deltaTime.value = Math.min(delta, 0.033);
    physics.uTime.value = simulationTime;

    const currentPistonX = getPistonXFromSurge();
    physics.pistonX.value = currentPistonX;
    pistonMesh.position.x = currentPistonX;

    controls.update();

    // --- 1. PHYSICS COMPUTE PIPELINE ---
    renderer.compute(physics.getClearGridPass());
    renderer.compute(physics.getGridPass());
    renderer.compute(physics.getBuildListPass());
    renderer.compute(physics.getComputePipeline());

    // --- 2. RENDER PIPELINE ---
    // First, render the background environment (tank, piston, etc.) to the screen
    renderer.render(scene, camera);

    // Then, let SSF handle the fluid. 
    // Right now, SSF will draw over the screen with the grayscale thickness map.
    ssf.render(renderer, camera);
  }
}

init();