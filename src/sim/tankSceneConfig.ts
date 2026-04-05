/**
 * Single source of truth for tank dimensions, piston motion, physics tuning, and scene alignment.
 * Visual meshes and GPU simulation both read from here.
 */

// =============================================================================
// Tank & visuals
// =============================================================================

/** Interior of the tank in world space (matches BoxGeometry + position). */
export const TANK = {
  /** Half-width in X (inner wall at ±halfWidth). */
  halfWidth: 25,
  /** Half-depth in Z (inner wall at ±halfDepth). */
  halfDepth: 15,
  floorY: 0,
  ceilingY: 10,
} as const;

/** Arguments for `THREE.BoxGeometry` and `tankMesh.position.y`. */
export const tankVisual = {
  width: TANK.halfWidth * 2,
  height: TANK.ceilingY - TANK.floorY,
  depth: TANK.halfDepth * 2,
  centerY: TANK.floorY + (TANK.ceilingY - TANK.floorY) / 2,
} as const;

// =============================================================================
// Piston (visual + physics uniform)
// =============================================================================

export const PISTON = {
  thickness: 1,
  /** `Math.max(0, sin(t * surgeAngularSpeed)) * surgeAmplitude` added to restX. */
  surgeAmplitude: 10,
  surgeAngularSpeed: 1.0,
  /** X position when surge factor is 0 (visual mesh + physics uniform). */
  restX: -24,
  /** Imparted X velocity when the piston catches a particle. */
  shoveVelocity: 15,
} as const;

// =============================================================================
// Recycling / respawn (far wall)
// =============================================================================

export const RECYCLE = {
  exitX: TANK.halfWidth,
  spawnX: -TANK.halfWidth,
  /** Base Y when a particle respawns at spawnX before swell sine. */
  spawnHeightBase: 1.5,
  /** Multiplier on `uTime` inside `sin(...)` for respawn height variation. */
  swellHeightFrequency: 2.0,
} as const;

// =============================================================================
// Reef obstacle (SDF + lift)
// =============================================================================

export const REEF = {
  center: { x: 5, y: 0, z: 0 },
  halfExtents: { x: 10, y: 0.5, z: 20 },
  liftFrontFaceThresholdX: -4,
  /** Extra upward velocity when hitting reef front (GPU adds to vel.y). */
  liftImpulse: 0.5,
} as const;

// =============================================================================
// Particle spawn (CPU init)
// =============================================================================

export const PARTICLE_SPAWN = {
  /** X range: [xMin, xMax) */
  xMin: -22,
  xMax: -8,
  yMin: 0.1,
  yMax: 1.1,
  /** Total Z span (symmetric around 0). */
  zSpan: 30,
} as const;

// =============================================================================
// SPH / pressure (kernel-sum density, not kg/m³)
// =============================================================================

export const PHYSICS_SPH = {
  /**
   * Target rest density ρ₀ in the same units as the density pass
   * (Σ poly6·(h²−r²)³ over neighbors). If too high vs actual sums, repulsion never turns on;
   * if too low, everything explodes apart.
   */
  restDensityKernel: 30.0,
  /** `pressure = stiffness * (ρ − ρ₀)` — higher = stiffer, snappier, easier to go unstable. */
  stiffness: 8.0,
  /** Scales outward push along `(pos_i − pos_j)` when `ρ > ρ₀`. */
  repulsionOverGain: 1.5,
  /** Scales inward pull along neighbor direction when `ρ < ρ₀` (cohesion). */
  cohesionUnderGain: 1.0,
  /** Vertical acceleration from `pressure / ρ₀` (buoyancy-style term). */
  buoyancyPressureGain: 3.5,
  /** Skip `normalize(delta)` when `|r|²` is below this (avoids NaNs). */
  neighborMinDistSq: 1e-8,
  viscosity: 0.8,
} as const;

// =============================================================================
// Gravity
// =============================================================================

export const PHYSICS_GRAVITY = {
  /** Downward acceleration (m/s² style world units). */
  y: -9.8,
} as const;

// =============================================================================
// Spatial hash grid (must cover the tank in world space)
// =============================================================================

export const PHYSICS_GRID = {
  resX: 64,
  resY: 32,
  resZ: 64,
  /** Smoothing length h; poly6 kernel and neighbor cutoff use this. */
  cellSize: 1.2,
  /** World-space offset added before `floor((pos + origin) / cellSize)`. */
  originX: 32,
  originY: 0,
  originZ: 32,
  /** Max steps walking each cell’s linked list (cost vs completeness). */
  neighborListMax: 64,
  /** 3×3×3 stencil for neighbor cells (fixed). */
  stencilCellCount: 27,
} as const;

/** Total uniform-grid cells (`resX * resY * resZ`) for clear pass and header buffers. */
export const PHYSICS_GRID_CELL_COUNT =
  PHYSICS_GRID.resX * PHYSICS_GRID.resY * PHYSICS_GRID.resZ;

// =============================================================================
// Collisions & contact damping (tank walls, floor, reef)
// =============================================================================

export const PHYSICS_COLLISION = {
  /** After floor clamp, if `y` is below this, treat as “on floor” for vertical velocity scale. */
  floorInteractionY: 0.01,
  /** Multiplies `vel.y` when near floor (negative = damped / inverted bounce). */
  floorVelocityYScale: -0.2,
  /** Multiplies `vel.x` after hitting +X tank wall. */
  wallVelocityScaleX: -0.5,
  /** Multiplies `vel.z` after hitting ±Z tank walls. */
  wallVelocityScaleZ: -0.5,
  /** Retains this fraction of tangent Z velocity after floor/reef contact (1 = no extra damping). */
  contactTangentZRetain: 0.82,
  /** Retains this fraction of tangent X velocity after floor/reef contact. */
  contactTangentXRetain: 0.95,
  /** Multiplies normal component of velocity after resolving contact (negative = inelastic). */
  contactNormalVelocityScale: -0.1,
} as const;

// =============================================================================
// GPU uniform initial values (can still be changed at runtime via `.value`)
// =============================================================================

export const PHYSICS_UNIFORM_DEFAULTS = {
  deltaTime: 0.016,
  swellAmplitude: 2.0,
  swellPeriod: 10.0,
  swellSpeed: 2.0,
} as const;

// =============================================================================
// Helpers
// =============================================================================

/**
 * Piston X for a given time — use for both `pistonMesh.position.x` and `physics.pistonX.value`.
 */
export function getPistonX(totalTime: number): number {
  const surge =
    Math.max(0, Math.sin(totalTime * PISTON.surgeAngularSpeed)) * PISTON.surgeAmplitude;
  return PISTON.restX + surge;
}
