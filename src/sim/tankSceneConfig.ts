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
  halfWidth: 20,
  /** Half-depth in Z (inner wall at ±halfDepth). */
  halfDepth: 6,
  floorY: 0,
  /** Taller column so the same particle count reads as deeper water. */
  ceilingY: 14,
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

const PISTON_THICKNESS = 0.4;

export const PISTON = {
  thickness: PISTON_THICKNESS,
  /** `Math.max(0, sin(t * surgeAngularSpeed)) * surgeAmplitude` added to restX. */
  surgeAmplitude: 3.5,
  surgeAngularSpeed: 1.0,
  /** Center X when surge factor is 0 — left face flush with inner wall at −halfWidth. */
  restX: -TANK.halfWidth + PISTON_THICKNESS * 0.5,
  /** Imparted X velocity when the piston catches a particle. */
  shoveVelocity: 5,
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
  /** Sits on the floor; top below the ceiling. */
  center: { x: 2, y: 0.35, z: 0 },
  /** X/Z flush with tank interior (no side gaps); Z span = 2·halfDepth. */
  halfExtents: { x: 4, y: 0.35, z: TANK.halfDepth },
  /** Local X (pos − reef center): west-of-center band that gets lift (matches old ratio ~). */
  liftFrontFaceThresholdX: -2,
  /** Extra upward velocity when hitting reef front (GPU adds to vel.y). */
  liftImpulse: 0.22,
} as const;

// =============================================================================
// Particle spawn (CPU init)
// =============================================================================

export const PARTICLE_SPAWN = {
  /** X range: [xMin, xMax) — between piston face and reef, with margin. */
  xMin: -TANK.halfWidth + PISTON.thickness + 0.08,
  xMax: REEF.center.x - REEF.halfExtents.x - 0.08,
  yMin: 0.08,
  yMax: 4.6,
  /** Total Z span (symmetric around 0), slightly inside ±halfDepth — tighter = higher N/V for same count. */
  zSpan: TANK.halfDepth * 2 - 0.28,
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
  /** Scales inward pull along neighbor direction when `ρ < ρ₀` (cohesion). Higher = fewer voids / stringy gaps. */
  cohesionUnderGain: 1.32,
  /** Vertical acceleration from `pressure / ρ₀` (buoyancy-style term). */
  buoyancyPressureGain: 3.5,
  /** Skip `normalize(delta)` when `|r|²` is below this (avoids NaNs). */
  neighborMinDistSq: 1e-8,
  /** Slightly higher damps shear and helps the mass read as one clump (too high = molasses). */
  viscosity: 1.05,
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
  /** Sized for TANK extents at cellSize 1.2 (keeps SPH neighbor counts stable). */
  resX: 24,
  resY: 16,
  resZ: 20,
  /** Smoothing length h; poly6 kernel and neighbor cutoff use this. */
  cellSize: 1.2,
  /** World-space offset added before `floor((pos + origin) / cellSize)`. */
  originX: 16,
  originY: 1,
  originZ: 12,
  /** Max steps walking each cell’s linked list (cost vs completeness). Higher helps dense 100k+ pools see enough neighbors. */
  neighborListMax: 112,
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
// Screen-space fluid (SSF) — thickness splats + bilateral blur + composite
// =============================================================================

export const GRAPHICS_SSF = {
  /**
   * Thickness pass resolution as a fraction of the drawable size (after DPR).
   * Lower = faster, chunkier edges; higher = sharper water silhouette, heavier GPU cost.
   * The splat pass temporarily sets the perspective camera aspect to `RTw/RTh` so projection
   * matches this buffer (avoids parallax slip vs the main scene when this differs from the canvas).
   */
  resolutionScale: 0.48,

  /**
   * Per-splat peak contribution written into the additive thickness RT (before blur).
   * With huge overlap + blur, keep low to avoid clipping float RT and “hot” beads.
   */
  splatPeak: 0.021,

  /**
   * `SpriteNodeMaterial` base scale. With `sizeAttenuation: false` in SSFManager, Three scales by view depth
   * so splats stay similar **screen size** — these numbers are smaller than the old world-fixed splats.
   */
  splatScaleX: 0.048,
  splatScaleY: 0.048,

  /**
   * `pow(1−r², p)` on the splat — **< 1** widens the shoulder so neighbors overlap before blur (fills RT gaps).
   */
  splatFalloffPower: 0.42,

  /**
   * Bilateral blur half-width in pixels (integer kernel size 2r+1).
   * Wider kernel fuses splat edges into a continuous sheet (cost grows as ~(2r+1)²).
   */
  blurRadius: 1,

  /**
   * Multiplier on `blurRadius²` in the spatial Gaussian denominator — **> 1** widens the kernel (soupier merge), **< 1** tightens.
   */
  blurSpatialScale: 2.62,

  /**
   * Bilateral range sigma in thickness units — how similar neighbors must be to blend.
   * Higher = more Gaussian-like smoothing in dense fluid, hides individual splat boundaries (also softens sharp silhouettes).
   */
  blurDepthThreshold: 0.82,

  /**
   * Scales blurred thickness before color/absorption (overall water density read).
   * Slightly lower after stronger blur/overlap so the mass doesn’t go uniformly opaque too fast.
   */
  thicknessDisplayGain: 0.29,

  /**
   * Beer–Lambert style opacity: alpha ≈ 1 − exp(−k × thickness).
   * A bit lower keeps merged regions from turning into a flat opaque slab too quickly.
   */
  absorption: 2.48,

  /** Deep / thick regions tint (hex). */
  waterColorDeep: 0x062a3d,

  /** Shallow / thin regions tint (hex). */
  waterColorShallow: 0x5dd4f0,

  /**
   * Fake ambient term on water (0–1 scale). Higher = flatter, less contrast; lower = darker hollows.
   */
  ambientStrength: 0.26,

  /**
   * Fake diffuse from screen-space thickness gradient (0–1). Lower = less “per-blob” shading, reads more uniform.
   */
  diffuseStrength: 0.16,

  /**
   * How much thickness gradient bends the fake normal. Lower = smoother, less lumpy lighting on the surface.
   */
  normalFromThicknessScale: 1.1,

  /** Fake directional light direction (view-ish space; normalized in shader). */
  lightDirX: 0.35,
  lightDirY: 0.75,
  lightDirZ: 0.3,

  /** Narrow specular lobe on top of diffuse (0–1). Lower = fewer sparkles on individual splats. */
  specularStrength: 0.1,
  specularPower: 36,
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
