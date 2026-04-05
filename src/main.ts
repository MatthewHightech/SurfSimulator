import * as THREE from 'three';
import { PointsNodeMaterial, WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PhysicsEngine } from './Physics/PhysicsEngine';
import { getPistonX, PISTON, REEF, tankVisual } from './sim/tankSceneConfig';

const PARTICLE_COUNT = 100000; // Let's go for 100k now

let physics: PhysicsEngine;
let particles: THREE.Points;

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
    // CRITICAL: Wait for the GPU to be ready
    await renderer.init();
    console.log("WebGPU Initialized.");

    physics = new PhysicsEngine(PARTICLE_COUNT);

    // 1. Temporary "High-Visibility" Material
    const testMaterial = new PointsNodeMaterial({
      color: 0x00ffff, // Bright Cyan
      size: 1,       // Large enough to see clearly
      sizeAttenuation: false
    });

    // 2. Link the Buffer
    testMaterial.positionNode = physics.getParticlePositionNode();

    // 3. Setup Geometry
    const geometry = new THREE.BufferGeometry();
    // Dummy attribute to define the "draw count"
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PARTICLE_COUNT * 3), 3));
    
    particles = new THREE.Points(geometry, testMaterial);
    particles.frustumCulled = false; 
    scene.add(particles);

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
  // Ensure physics is ready before updating
  if (physics) {
    physics.deltaTime.value = Math.min(delta, 0.033);
    controls.update();

    physics.uTime.value = totalTime;

    const currentPistonX = getPistonX(totalTime);
    physics.pistonX.value = currentPistonX;
    pistonMesh.position.x = currentPistonX;

    // --- THE SPH PIPELINE ---
    // 1. Clear the "Post Office" (Grid Headers)
    renderer.compute(physics.getClearGridPass());

    // 2. Hash: Every particle finds its 3D cell
    renderer.compute(physics.getGridPass());

    // 3. Build List: Link particles in the same cells
    renderer.compute(physics.getBuildListPass());

    // 4. Density: Particles look at neighbors to see how "squashed" they are
    renderer.compute(physics.getDensityPass());

    // Run compute then render
    renderer.compute(physics.getComputePipeline());
    renderer.render(scene, camera);
  }
}

init();