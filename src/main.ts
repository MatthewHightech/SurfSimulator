import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PhysicsEngine } from './Physics/PhysicsEngine';
import { SSFManager } from './Graphics/SSFManager'; // Import your new manager
import { getPistonX, PISTON, REEF, tankVisual } from './sim/tankSceneConfig';

/** Default raised after merging density+forces into one compute pass (~2× less neighbor work). */
const PARTICLE_COUNT = 220_000;

let physics: PhysicsEngine;
let ssf: SSFManager;

const renderer = new WebGPURenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(20, 20, 20);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

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

async function init() {
  try {
    await renderer.init();
    console.log("WebGPU Initialized.");

    physics = new PhysicsEngine(PARTICLE_COUNT);
    
    ssf = new SSFManager(renderer, physics);

    animate();
  } catch (e) {
    console.error("WebGPU error:", e);
  }
}

function animate() {
  requestAnimationFrame(animate);
  clock.update();
  
  const delta = clock.getDelta();
  const totalTime = clock.getElapsed();

  if (physics && ssf) {
    // Update Uniforms
    physics.deltaTime.value = Math.min(delta, 0.033);
    physics.uTime.value = totalTime;
    
    const currentPistonX = getPistonX(totalTime);
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