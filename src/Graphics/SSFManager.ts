import {
  Fn,
  float,
  vec2,
  vec3,
  vec4,
  max,
  abs,
  uv,
  texture,
  uniform,
  exp,
  pow,
  mix,
  clamp,
  normalize,
  color,
  dFdx,
  dFdy,
  screenUV,
} from 'three/tsl';
import { MeshBasicNodeMaterial, QuadMesh, SpriteNodeMaterial, WebGPURenderer } from 'three/webgpu';
import * as THREE from 'three';

import { PhysicsEngine } from '../Physics/PhysicsEngine';
import { GRAPHICS_SSF as SSF } from '../sim/tankSceneConfig';

function createThicknessBlob(splatPeak: number, splatFalloffPower: number) {
  return Fn(() => {
    const half = float(0.5);
    const coord = uv().sub(vec2(half, half)).mul(2.0);
    const r2 = coord.dot(coord);
    const falloff = max(float(0), float(1.0).sub(r2));
    const shaped = pow(falloff, float(splatFalloffPower));
    return vec4(shaped.mul(float(splatPeak)), 0, 0, 1);
  });
}

export class SSFManager {
  private thicknessTarget: THREE.WebGLRenderTarget;
  private thicknessMaterial: SpriteNodeMaterial;
  private compositeMaterial: MeshBasicNodeMaterial;
  /** Fullscreen pass — uses engine QuadMesh + internal ortho camera (WebGPU-aligned). */
  private finalQuad: QuadMesh;
  private splatMesh: THREE.InstancedMesh;

  private readonly blurInvTexel = uniform(new THREE.Vector2());

  constructor(renderer: WebGPURenderer, physics: PhysicsEngine) {
    const size = new THREE.Vector2();
    renderer.getSize(size);
    const dpr = renderer.getPixelRatio();

    const rw = Math.max(1, Math.floor(size.x * dpr * SSF.resolutionScale));
    const rh = Math.max(1, Math.floor(size.y * dpr * SSF.resolutionScale));

    this.thicknessTarget = new THREE.WebGLRenderTarget(rw, rh, {
      type: THREE.FloatType,
      format: THREE.RedFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });

    this.blurInvTexel.value.set(1 / rw, 1 / rh);

    const particleCount = physics.getParticleCount();

    this.thicknessMaterial = new SpriteNodeMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      /**
       * `true` shrinks distant splats on screen → visible gaps and “lumps” at oblique views.
       * `false` keeps a more even screen footprint so overlap matches particle density (see `splatScale*`).
       */
      sizeAttenuation: false,
    });
    this.thicknessMaterial.colorNode = createThicknessBlob(SSF.splatPeak, SSF.splatFalloffPower)();
    this.thicknessMaterial.positionNode = physics.getPositionBuffer().toAttribute().xyz;
    this.thicknessMaterial.scaleNode = vec2(SSF.splatScaleX, SSF.splatScaleY);

    const quadGeo = new THREE.PlaneGeometry(1, 1);
    this.splatMesh = new THREE.InstancedMesh(quadGeo, this.thicknessMaterial, particleCount);
    // Must stay false: positions come from the node graph, not instance matrices, so bounds stay tiny.
    this.splatMesh.frustumCulled = false;

    const id = new THREE.Matrix4();
    for (let i = 0; i < particleCount; i++) {
      this.splatMesh.setMatrixAt(i, id);
    }
    this.splatMesh.instanceMatrix.needsUpdate = true;

    this.compositeMaterial = new MeshBasicNodeMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NormalBlending,
      toneMapped: true,
    });
    this.compositeMaterial.colorNode = this.createCompositeFluidNode();

    this.finalQuad = new QuadMesh(this.compositeMaterial);
  }

  private createCompositeFluidNode() {
    const blurR = SSF.blurRadius;
    const eps = float(0.0001);

    return Fn(() => {
      const tex = texture(this.thicknessTarget.texture);
      const invSize = this.blurInvTexel;

      // Mesh `uv()` on a plane does not match framebuffer coordinates; `screenUV` tracks the
      // same pixels as the main scene so the fluid lines up when the camera orbits.
      const centerUV = screenUV;
      const centerThickness = tex.sample(centerUV).r;

      const sum = float(0.0).toVar();
      const wSum = float(0.0).toVar();

      const blurDepthThreshold = float(SSF.blurDepthThreshold);

      for (let x = -blurR; x <= blurR; x++) {
        for (let y = -blurR; y <= blurR; y++) {
          const offset = vec2(float(x), float(y)).mul(invSize);
          const sampleThickness = tex.sample(centerUV.add(offset)).r;

          const distSq = float(x * x + y * y);
          const spatialDenom = float(blurR * blurR).mul(float(SSF.blurSpatialScale));
          const spatialW = exp(distSq.negate().div(spatialDenom));

          const diff = abs(centerThickness.sub(sampleThickness));
          const rangeW = exp(diff.mul(diff).negate().div(blurDepthThreshold.add(eps)));

          const finalW = spatialW.mul(rangeW);
          sum.addAssign(sampleThickness.mul(finalW));
          wSum.addAssign(finalW);
        }
      }

      const tRaw = sum.div(max(wSum, float(0.00001)));
      const t = tRaw.mul(float(SSF.thicknessDisplayGain));

      const t4 = vec4(t, float(0), float(0), float(0));
      const ddx = dFdx(t4).x;
      const ddy = dFdy(t4).x;
      const nz = float(SSF.normalFromThicknessScale);
      const N = normalize(vec3(ddx.negate().mul(nz), ddy.negate().mul(nz), float(1.0)));
      const L = normalize(vec3(SSF.lightDirX, SSF.lightDirY, SSF.lightDirZ));
      const V = vec3(0, 0, 1);
      const H = normalize(L.add(V));
      const ndotl = max(float(0), N.dot(L));
      const ndoth = max(float(0), N.dot(H));
      const amb = float(SSF.ambientStrength);
      const diffuseTerm = float(SSF.diffuseStrength).mul(ndotl);
      const spec = pow(ndoth, float(SSF.specularPower)).mul(float(SSF.specularStrength));
      const luminance = amb.add(diffuseTerm).add(spec);

      const deep = color(SSF.waterColorDeep);
      const shallow = color(SSF.waterColorShallow);
      const tn = clamp(t, float(0), float(8));
      const mixW = pow(tn.div(tn.add(float(0.35))), float(0.55));
      const waterRgb = mix(deep, shallow, mixW);
      const lit = waterRgb.mul(luminance);

      const alpha = float(1).sub(exp(t.negate().mul(float(SSF.absorption))));
      const a = clamp(alpha, float(0), float(0.96));

      return vec4(lit, a);
    })();
  }

  public render(renderer: WebGPURenderer, camera: THREE.Camera) {
    const rt = this.thicknessTarget;
    const rw = rt.width;
    const rh = rt.height;

    // Perspective matrix must use the RT aspect (after floor), not only the canvas aspect, or
    // splats project on a different frustum than the main pass → sliding parallax vs the tank.
    const persp = camera instanceof THREE.PerspectiveCamera ? camera : null;
    const savedAspect = persp?.aspect ?? 1;
    if (persp) {
      persp.aspect = rw / rh;
      persp.updateProjectionMatrix();
    }

    renderer.setRenderTarget(rt);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(this.splatMesh, camera);

    if (persp) {
      persp.aspect = savedAspect;
      persp.updateProjectionMatrix();
    }

    renderer.setRenderTarget(null);

    // Default `autoClear` clears color/depth before every render. A second render in the same
    // frame would wipe the main scene — only the water quad would remain.
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    this.finalQuad.render(renderer);
    renderer.autoClear = prevAutoClear;
  }
}
