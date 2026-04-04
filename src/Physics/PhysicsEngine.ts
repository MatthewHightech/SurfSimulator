import {
  Fn,
  float,
  instanceIndex,
  vertexIndex,
  vec3,
  uniform,
  select,
  storage,
  max,
  min,
  normalize,
  If,
  abs,
  length,
  sin,
} from 'three/tsl';
import { StorageInstancedBufferAttribute } from 'three/webgpu';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

export class PhysicsEngine {
  private readonly numParticles: number;
  private readonly positionBuffer: StorageBufferNode<'vec4'>;
  private readonly velocityBuffer: StorageBufferNode<'vec4'>;
  private readonly computeNode: ComputeNode;

  public deltaTime = uniform(0.016);

  public uTime = uniform(0.0);
  public swellAmplitude = uniform(2.0);
  public swellPeriod = uniform(10.0);
  public swellSpeed = uniform(5.0);

  constructor(numParticles: number) {
    this.numParticles = numParticles;

    const posAttr = new StorageInstancedBufferAttribute(numParticles, 4);
    const velAttr = new StorageInstancedBufferAttribute(numParticles, 4);

    // Initial "Block" of water on the left side of the screen
    for (let i = 0; i < numParticles; i++) {
      posAttr.setXYZW(
        i, 
        -25 + Math.random() * 10, // Start on the left (-25 to -15)
        1 + Math.random() * 5,    // Close to the ground
        (Math.random() - 0.5) * 15, 
        1
      );
      velAttr.setXYZW(i, 0, 0, 0, 0);
    }

    this.positionBuffer = storage(posAttr, 'vec4', numParticles);
    this.velocityBuffer = storage(velAttr, 'vec4', numParticles);
    this.computeNode = this.createComputePipeline();
  }

  private createComputePipeline(): ComputeNode {
    const computeLogic = Fn(() => {
      const pos = this.positionBuffer.element(instanceIndex).toVar();
      const vel = this.velocityBuffer.element(instanceIndex).toVar();
    
      // 1. FORCES: Increased current to help particles "climb" the reef
      const gravity = vec3(0, -9.8, 0);
      const current = vec3(4.0, 0, 0); // Bumped from 1.5 to 4.0
      vel.xyz.addAssign(gravity.add(current).mul(this.deltaTime));
      pos.xyz.addAssign(vel.xyz.mul(this.deltaTime));
    
      // 2. THE REEF (Wider and offset)
      const reefCenter = vec3(0.0, -0.5, 0.0); // Offset to the right, centered at Y=2
      const reefHalfExtents = vec3(10.0, 1.0, 50.0); // 4 units high total
      
      const p = pos.xyz.sub(reefCenter);
      const d = abs(p).sub(reefHalfExtents);
      const distToReef = length(max(d, float(0.0))).add(min(max(d.x, max(d.y, d.z)), float(0.0)));
      
      const distToFloor = pos.y;
      const worldDist = min(distToFloor, distToReef);
      const hitWorld = worldDist.lessThan(0.0);
    
      If(hitWorld, () => {
        const isFloorHit = distToFloor.lessThan(distToReef);
        const normal = select(isFloorHit, vec3(0, 1, 0), normalize(p));
    
        pos.xyz.assign(pos.xyz.sub(normal.mul(worldDist)));
        
        // COLLISION RESPONSE
        const dotProduct = vel.xyz.dot(normal);
        const vNormal = normal.mul(dotProduct);
        const vTangent = vel.xyz.sub(vNormal);
        
        // SENIOR MOVE: Add "Directional Friction"
        // We dampen Z-velocity (sideways) heavily, but keep X-velocity (forward) fast
        vTangent.z.mulAssign(0.1); // Kill the "sliding around" momentum
        vTangent.x.mulAssign(0.99); // Keep the "sliding over" momentum
        
        vel.xyz.assign(vTangent.add(vNormal.mul(-0.1)));
    
        // THE "LIFT" HACK: 
        // If we're hitting the FRONT face of the reef, give a tiny Y-boost 
        // to simulate the upward draw of a wave.
        const isFrontFace = p.x.lessThan(-7.0); 
        If(isFrontFace, () => {
           vel.y.addAssign(float(2.0));
        });
      });
    
      // 3. RECYCLING
      const isPastEnd = pos.x.greaterThan(25.0);
      If(isPastEnd, () => {
        pos.x.assign(-25.0);
        // Height variation to create "set" pulses
        pos.y.assign(float(1.5).add(sin(this.uTime.mul(2.0)).mul(this.swellAmplitude)));
        vel.x.assign(this.swellSpeed);
        vel.y.assign(0);
      });
    
      this.positionBuffer.element(instanceIndex).assign(pos);
      this.velocityBuffer.element(instanceIndex).assign(vel);
    });

    return computeLogic().compute(this.numParticles);
  }

  getComputeNode(): ComputeNode {
    return this.computeNode;
  }

  // Helper for the Mesh/Points to find the data
  public getPositionBuffer() {
    return this.positionBuffer;
  }

  public getParticlePositionNode() {
    return this.positionBuffer.element(vertexIndex).xyz;  }
}
