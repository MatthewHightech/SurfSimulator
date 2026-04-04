import * as THREE from 'three';
import { PointsNodeMaterial, WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PhysicsEngine } from './Physics/PhysicsEngine';

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
scene.add(new THREE.GridHelper(50, 50, 0x222222, 0x111111));
scene.add(new THREE.AxesHelper(5));

// Placeholder Reef

scene.add(new THREE.AmbientLight(0xffffff, 1));

// add directional light
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(10, 10, 10);
scene.add(directionalLight);

  const clock = new THREE.Clock();


async function init() {
  try {
    // CRITICAL: Wait for the GPU to be ready
    await renderer.init();
    console.log("WebGPU Initialized.");

    physics = new PhysicsEngine(PARTICLE_COUNT);

    // 1. Temporary "High-Visibility" Material
    const testMaterial = new PointsNodeMaterial({
      color: 0x00ffff, // Bright Cyan
      size: 0.5,       // Large enough to see clearly
      sizeAttenuation: true
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
  const delta = clock.getDelta();
  // Ensure physics is ready before updating
  if (physics) {
    physics.deltaTime.value = Math.min(delta, 0.033);
    controls.update();

    const elapsedTime = clock.getElapsedTime();
    physics.uTime.value = elapsedTime;

    // Run compute then render
    renderer.compute(physics.getComputeNode());
    renderer.render(scene, camera);
  }
}

init();