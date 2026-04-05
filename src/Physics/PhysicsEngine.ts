import {
  Fn,
  float,
  int,
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
  uint,
  floor,
  clamp,
  pow,
  lengthSq,
  Break,
  Loop,
} from 'three/tsl';
import { StorageInstancedBufferAttribute } from 'three/webgpu';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type Node from 'three/src/nodes/core/Node.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import {
  PARTICLE_SPAWN,
  PISTON,
  RECYCLE,
  REEF,
  TANK,
  PHYSICS_SPH,
  PHYSICS_GRAVITY,
  PHYSICS_GRID,
  PHYSICS_GRID_CELL_COUNT,
  PHYSICS_COLLISION,
  PHYSICS_UNIFORM_DEFAULTS,
} from '../sim/tankSceneConfig';

export class PhysicsEngine {
  private readonly numParticles: number;
  private readonly positionBuffer: StorageBufferNode<'vec4'>;
  private readonly velocityBuffer: StorageBufferNode<'vec4'>;
  private readonly gridBuffer: StorageBufferNode<'uint'>; 
  private readonly densityBuffer: StorageBufferNode<'float'>;
  private readonly gridHeaderBuffer: StorageBufferNode<'uint'>;
  private readonly gridNextBuffer: StorageBufferNode<'uint'>;

  public deltaTime = uniform(PHYSICS_UNIFORM_DEFAULTS.deltaTime);

  public uTime = uniform(0.0);
  public swellAmplitude = uniform(PHYSICS_UNIFORM_DEFAULTS.swellAmplitude);
  public swellPeriod = uniform(PHYSICS_UNIFORM_DEFAULTS.swellPeriod);
  public swellSpeed = uniform(PHYSICS_UNIFORM_DEFAULTS.swellSpeed);

  private readonly gridRes = vec3(PHYSICS_GRID.resX, PHYSICS_GRID.resY, PHYSICS_GRID.resZ);
  private readonly cellSize = float(PHYSICS_GRID.cellSize);

  public pistonX = uniform(PISTON.restX);
  /** maxX, unused Y slot, halfDepth for Z walls — matches {@link TANK}. */
  public tankBounds = vec3(TANK.halfWidth, TANK.ceilingY, TANK.halfDepth);

  // Store the "Baked" Compute Nodes
  private clearGridNode!: ComputeNode;
  private gridPassNode!: ComputeNode;
  private buildListNode!: ComputeNode;
  private densityPassNode!: ComputeNode;
  private computePipelineNode!: ComputeNode;

  constructor(numParticles: number) {
    this.numParticles = numParticles;
    const cellCount = PHYSICS_GRID_CELL_COUNT;

    // 1. DENSITY BUFFER (Per-particle float)
    const densityAttr = new StorageInstancedBufferAttribute(numParticles, 1);
    const gridAttr = new StorageInstancedBufferAttribute(numParticles, 1);
    this.densityBuffer = storage(densityAttr, 'float', numParticles);
    this.gridBuffer = storage(gridAttr, 'uint', numParticles);

    const headerAttr = new StorageInstancedBufferAttribute(cellCount, 1);
    const nextAttr = new StorageInstancedBufferAttribute(numParticles, 1);
    // Plain `uint` storage (not `atomic<>`): see createBuildListPass note below.
    this.gridHeaderBuffer = storage(headerAttr, 'uint', cellCount);
    this.gridNextBuffer = storage(nextAttr, 'uint', numParticles);

    const posAttr = new StorageInstancedBufferAttribute(numParticles, 4);
    const velAttr = new StorageInstancedBufferAttribute(numParticles, 4);
    this.positionBuffer = storage(posAttr, 'vec4', numParticles);
    this.velocityBuffer = storage(velAttr, 'vec4', numParticles);

    this.clearGridNode = this.createClearGridPass();
    this.gridPassNode = this.createGridPass();
    this.buildListNode = this.createBuildListPass();
    this.densityPassNode = this.createDensityPass();
    this.computePipelineNode = this.createComputePipeline();

    const { xMin, xMax, yMin, yMax, zSpan } = PARTICLE_SPAWN;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    for (let i = 0; i < numParticles; i++) {
      posAttr.setXYZW(
        i,
        xMin + Math.random() * xRange,
        yMin + Math.random() * yRange,
        (Math.random() - 0.5) * zSpan,
        1,
      );
      velAttr.setXYZW(i, 0, 0, 0, 0);
    }
  }

  private createComputePipeline(): ComputeNode {
    const computeLogic = Fn(() => {
      const pos = this.positionBuffer.element(instanceIndex).toVar();
      const vel = this.velocityBuffer.element(instanceIndex).toVar();

      const rho = this.densityBuffer.element(instanceIndex);
      const rho0 = float(PHYSICS_SPH.restDensityKernel);
      const stiffness = float(PHYSICS_SPH.stiffness);

      const pressure = stiffness.mul(rho.sub(rho0));
      const over = max(rho.sub(rho0), float(0));
      const under = max(rho0.sub(rho), float(0));

      // IMPORTANT: A vec3(0, f(p), 0) "pressure" only pushes vertically. Under gravity
      // everything hits the floor and spreads in XZ with no lateral repulsion → a 2D pancake.
      // Real SPH uses −∇p (3D). We approximate lateral + vertical repulsion by summing
      // over neighbors (same 3³ stencil as density) along separation directions.
      const gridOff = vec3(PHYSICS_GRID.originX, PHYSICS_GRID.originY, PHYSICS_GRID.originZ);
      const gc = floor(pos.xyz.add(gridOff).div(this.cellSize));
      const resX = uint(this.gridRes.x);
      const resY = uint(this.gridRes.y);
      const h = this.cellSize;
      const h2 = h.mul(h);
      const repulsion = vec3(0, 0, 0).toVar();

      Loop(
        { start: int(0), end: int(PHYSICS_GRID.stencilCellCount), type: 'int', condition: '<' },
        ({ i }) => {
        const t = float(i);
        const ox = t.mod(3).floor().sub(1);
        const oy = floor(t.div(3)).mod(3).floor().sub(1);
        const oz = floor(t.div(9)).mod(3).floor().sub(1);
        const nx = clamp(gc.x.add(ox), float(0), float(this.gridRes.x.sub(1))).toUint();
        const ny = clamp(gc.y.add(oy), float(0), float(this.gridRes.y.sub(1))).toUint();
        const nz = clamp(gc.z.add(oz), float(0), float(this.gridRes.z.sub(1))).toUint();
        const neighHash = nx.add(ny.mul(resX)).add(nz.mul(resX).mul(resY));
        const neighborIdx = this.gridHeaderBuffer.element(neighHash).toVar();

        Loop({ start: uint(0), end: uint(PHYSICS_GRID.neighborListMax) }, () => {
          If(neighborIdx.equal(uint(0xffffffff)), () => Break());
          const posJ = this.positionBuffer.element(neighborIdx).xyz;
          const delta = pos.xyz.sub(posJ);
          const r2 = lengthSq(delta);
          If(
            r2.lessThan(h2).and(r2.greaterThan(float(PHYSICS_SPH.neighborMinDistSq))),
            () => {
            const dir = normalize(delta);
            repulsion.addAssign(dir.mul(over).div(rho0).mul(float(PHYSICS_SPH.repulsionOverGain)));
            repulsion.addAssign(
              dir.negate().mul(under).div(rho0).mul(float(PHYSICS_SPH.cohesionUnderGain)),
            );

            // This dampens the velocity difference between particle I and neighbor J
            const velJ = this.velocityBuffer.element(neighborIdx).xyz;
            const relativeVel = vel.xyz.sub(velJ);
            const vDotR = relativeVel.dot(delta);
            
            // If particles are moving toward each other (vDotR < 0), apply friction
            If(vDotR.lessThan(0.0), () => {
                const viscIntensity = float(PHYSICS_SPH.viscosity); // Tune this to make water "thicker"
                const friction = delta.mul(vDotR).div(r2.add(float(0.01))).mul(viscIntensity);
                repulsion.subAssign(friction); 
            });
          },
          );
          neighborIdx.assign(this.gridNextBuffer.element(neighborIdx));
        });
      },
      );

      const gravity = vec3(0, PHYSICS_GRAVITY.y, 0);
      const buoyancy = vec3(0, pressure.div(rho0).mul(PHYSICS_SPH.buoyancyPressureGain), 0);
      const totalAccel = gravity.add(buoyancy).add(repulsion);
      vel.xyz.addAssign(totalAccel.mul(this.deltaTime));
      pos.xyz.addAssign(vel.xyz.mul(this.deltaTime));

      const behindPiston = pos.x.lessThan(this.pistonX);
      pos.x.assign(select(behindPiston, this.pistonX, pos.x));
      vel.x.assign(select(behindPiston, float(PISTON.shoveVelocity), vel.x));

      pos.y.assign(max(pos.y, float(TANK.floorY)));
      vel.y.assign(
        select(
          pos.y.lessThan(float(PHYSICS_COLLISION.floorInteractionY)),
          vel.y.mul(float(PHYSICS_COLLISION.floorVelocityYScale)),
          vel.y,
        ),
      );

      const outZ = abs(pos.z).greaterThan(this.tankBounds.z);
      pos.z.assign(select(outZ, this.tankBounds.z.mul(pos.z.sign()), pos.z));
      vel.z.assign(select(outZ, vel.z.mul(float(PHYSICS_COLLISION.wallVelocityScaleZ)), vel.z));

      const outX = pos.x.greaterThan(this.tankBounds.x);
      pos.x.assign(select(outX, this.tankBounds.x, pos.x));
      vel.x.assign(select(outX, vel.x.mul(float(PHYSICS_COLLISION.wallVelocityScaleX)), vel.x));

      const reefCenter = vec3(REEF.center.x, REEF.center.y, REEF.center.z);
      const reefHalfExtents = vec3(
        REEF.halfExtents.x,
        REEF.halfExtents.y,
        REEF.halfExtents.z,
      );

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

        const dotProduct = vel.xyz.dot(normal);
        const vNormal = normal.mul(dotProduct);
        const vTangent = vel.xyz.sub(vNormal);

        vTangent.z.mulAssign(float(PHYSICS_COLLISION.contactTangentZRetain));
        vTangent.x.mulAssign(float(PHYSICS_COLLISION.contactTangentXRetain));

        vel.xyz.assign(
          vTangent.add(vNormal.mul(float(PHYSICS_COLLISION.contactNormalVelocityScale))),
        );

        const isFrontFace = p.x.lessThan(float(REEF.liftFrontFaceThresholdX));
        If(isFrontFace, () => {
          vel.y.addAssign(float(REEF.liftImpulse));
        });
      });

      const isPastEnd = pos.x.greaterThan(float(RECYCLE.exitX));
      If(isPastEnd, () => {
        pos.x.assign(float(RECYCLE.spawnX));
        pos.y.assign(
          float(RECYCLE.spawnHeightBase).add(
            sin(this.uTime.mul(float(RECYCLE.swellHeightFrequency))).mul(this.swellAmplitude),
          ),
        );
        vel.x.assign(this.swellSpeed);
        vel.y.assign(0);
      });

      this.positionBuffer.element(instanceIndex).assign(pos);
      this.velocityBuffer.element(instanceIndex).assign(vel);
    });

    return computeLogic().compute(this.numParticles);
  }

  getPositionBuffer() {
    return this.positionBuffer;
  }

  getParticlePositionNode(): Node {
    return this.positionBuffer.element(vertexIndex).xyz;
  }

  private createDensityPass() {
    return Fn(() => {
      const posI = this.positionBuffer.element(instanceIndex).xyz;
      const h = this.cellSize;
      const h2 = h.mul(h);
      const density = float(0.0).toVar();

      const poly6 = float(315.0).div(float(64.0).mul(Math.PI).mul(pow(h, 9)));

      // Neighbors within radius h usually span a 3×3×3 block of cells. Using only the
      // home cell makes ρ far too small and breaks pressure — another pancake contributor.
      const gridOffset = vec3(PHYSICS_GRID.originX, PHYSICS_GRID.originY, PHYSICS_GRID.originZ);
      const gc = floor(posI.add(gridOffset).div(this.cellSize));
      const resX = uint(this.gridRes.x);
      const resY = uint(this.gridRes.y);

      Loop(
        { start: int(0), end: int(PHYSICS_GRID.stencilCellCount), type: 'int', condition: '<' },
        ({ i }) => {
        const t = float(i);
        const ox = t.mod(3).floor().sub(1);
        const oy = floor(t.div(3)).mod(3).floor().sub(1);
        const oz = floor(t.div(9)).mod(3).floor().sub(1);
        const nx = clamp(gc.x.add(ox), float(0), float(this.gridRes.x.sub(1))).toUint();
        const ny = clamp(gc.y.add(oy), float(0), float(this.gridRes.y.sub(1))).toUint();
        const nz = clamp(gc.z.add(oz), float(0), float(this.gridRes.z.sub(1))).toUint();
        const neighHash = nx.add(ny.mul(resX)).add(nz.mul(resX).mul(resY));

        const neighborIdx = this.gridHeaderBuffer.element(neighHash).toVar();

        Loop({ start: uint(0), end: uint(PHYSICS_GRID.neighborListMax) }, () => {
          If(neighborIdx.equal(uint(0xffffffff)), () => Break());

          const posJ = this.positionBuffer.element(neighborIdx).xyz;
          const r2 = lengthSq(posI.sub(posJ));

          If(r2.lessThan(h2), () => {
            const distTerm = h2.sub(r2);
            density.addAssign(poly6.mul(pow(distTerm, 3)));
          });

          neighborIdx.assign(this.gridNextBuffer.element(neighborIdx));
        });
      },
      );

      this.densityBuffer.element(instanceIndex).assign(density);
    })().compute(this.numParticles);
  }

  private createGridPass() {
    return Fn(() => {
      const pos = this.positionBuffer.element(instanceIndex).xyz;
      
      // 1. Grid Coordinate (This is a vec3 of floats)
      const gridOffset = vec3(PHYSICS_GRID.originX, PHYSICS_GRID.originY, PHYSICS_GRID.originZ);
      const gridCoord = floor(pos.add(gridOffset).div(this.cellSize));
  
      // 2. SENIOR FIX: Clamp as float, then cast to Uint
      // Notice we use float(0) and float(this.gridRes.x - 1)
      const x = clamp(gridCoord.x, float(0), float(this.gridRes.x.sub(1))).toUint();
      const y = clamp(gridCoord.y, float(0), float(this.gridRes.y.sub(1))).toUint();
      const z = clamp(gridCoord.z, float(0), float(this.gridRes.z.sub(1))).toUint();
  
      // 3. Compute the 1D Hash
      const resX = uint(this.gridRes.x);
      const resY = uint(this.gridRes.y);
      const hash = x.add(y.mul(resX)).add(z.mul(resX).mul(resY));
  
      // 4. Store it
      this.gridBuffer.element(instanceIndex).assign(hash);
    })().compute(this.numParticles);
  }


  public createBuildListPass() {
    return Fn(() => {
      const hash = this.gridBuffer.element(instanceIndex);
      const particleIdx = instanceIndex.toUint();
      const prevHead = this.gridHeaderBuffer.element(hash).toVar();
      this.gridHeaderBuffer.element(hash).assign(particleIdx);
      this.gridNextBuffer.element(instanceIndex).assign(prevHead);
    })().compute(this.numParticles);
  }

  public createClearGridPass() {
    return Fn(() => {
      this.gridHeaderBuffer.element(instanceIndex).assign(uint(0xffffffff));
    })().compute(PHYSICS_GRID_CELL_COUNT);
  }

  public getClearGridPass() { return this.clearGridNode; }
  public getGridPass() { return this.gridPassNode; }
  public getBuildListPass() { return this.buildListNode; }
  public getDensityPass() { return this.densityPassNode; }
  public getComputePipeline() { return this.computePipelineNode; }

}
