/**
 * Ambient Canvas Particle & Grid Mesh Background
 * Built adhering to `optimize-web-animations` & `dark-glass-clean-layout` skills.
 * Features IntersectionObserver & document.visibilityState pause gating to eliminate CPU/GPU overhead offscreen.
 */
class AmbientBackground {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.particleCount = 45;
    this.animFrameId = null;
    this.isActive = false;

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Create particle nodes
    this.particles = [];
    for (let i = 0; i < this.particleCount; i++) {
      this.particles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        radius: Math.random() * 1.5 + 1.0,
        alpha: Math.random() * 0.4 + 0.1
      });
    }

    // Optimization: Pause animation loop when offscreen or tab inactive
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !document.hidden) {
          this.start();
        } else {
          this.stop();
        }
      });
    }, { threshold: 0.05 });

    this.observer.observe(this.canvas);

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stop();
      } else {
        this.start();
      }
    });

    this.start();
  }

  resize() {
    this.width = this.canvas.width = window.innerWidth;
    this.height = this.canvas.height = window.innerHeight;
  }

  start() {
    if (!this.isActive) {
      this.isActive = true;
      this.loop();
    }
  }

  stop() {
    if (this.isActive) {
      this.isActive = false;
      if (this.animFrameId) {
        cancelAnimationFrame(this.animFrameId);
        this.animFrameId = null;
      }
    }
  }

  loop() {
    if (!this.isActive) return;

    this.ctx.clearRect(0, 0, this.width, this.height);

    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const pColor = isDark ? '56, 189, 248' : '2, 132, 199'; // Sky blue / Opta accent
    const lColor = isDark ? '51, 65, 85' : '229, 229, 224';

    // Update & draw particles
    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = this.width;
      if (p.x > this.width) p.x = 0;
      if (p.y < 0) p.y = this.height;
      if (p.y > this.height) p.y = 0;

      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = `rgba(${pColor}, ${p.alpha})`;
      this.ctx.fill();

      // Draw faint connecting lines
      for (let j = i + 1; j < this.particles.length; j++) {
        let p2 = this.particles[j];
        let dx = p.x - p2.x;
        let dy = p.y - p2.y;
        let dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 140) {
          this.ctx.beginPath();
          this.ctx.moveTo(p.x, p.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.strokeStyle = `rgba(${lColor}, ${(1 - dist / 140) * 0.25})`;
          this.ctx.lineWidth = 0.8;
          this.ctx.stroke();
        }
      }
    }

    this.animFrameId = requestAnimationFrame(() => this.loop());
  }
}

// Global instance launcher
window.addEventListener('DOMContentLoaded', () => {
  window.gAmbientBg = new AmbientBackground('ambient-canvas');
});
