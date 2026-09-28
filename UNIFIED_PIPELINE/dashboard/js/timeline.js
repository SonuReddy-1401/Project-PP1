/**
 * Synchronized Timeline & Animation Player Engine
 * Dynamic clock parsing for any input match start offset and clip duration.
 */
class TimelinePlayer {
  constructor() {
    this.totalFrames = 1000;
    this.fps = 25.0;
    this.startOffsetSec = 0.0;
    this.currentFrame = 0;
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.animationTimer = null;
    this.callbacks = [];
  }

  init(totalFrames, fps, startOffsetSec) {
    this.totalFrames = totalFrames || 1000;
    this.fps = fps || 25.0;
    this.startOffsetSec = startOffsetSec || 0.0;
    this.currentFrame = 0;
    this.isPlaying = false;

    this.bindEvents();
    this.updateUI();
  }

  addFrameCallback(cb) {
    this.callbacks.push(cb);
  }

  bindEvents() {
    const playBtn = document.getElementById('btn-play-pause');
    const slider = document.getElementById('timeline-scrubber');
    const speedBtns = document.querySelectorAll('.speed-btn');

    if (playBtn) {
      playBtn.onclick = () => this.togglePlay();
    }

    if (slider) {
      slider.max = this.totalFrames - 1;
      slider.oninput = (e) => {
        this.seekToFrame(parseInt(e.target.value, 10));
      };
    }

    speedBtns.forEach(btn => {
      btn.onclick = (e) => {
        speedBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.playbackSpeed = parseFloat(e.target.dataset.speed || 1.0);
        if (this.isPlaying) {
          this.pause();
          this.play();
        }
      };
    });
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    this.isPlaying = true;
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) playBtn.innerText = '⏸';

    const intervalMs = (1000.0 / this.fps) / this.playbackSpeed;
    this.animationTimer = setInterval(() => {
      this.currentFrame++;
      if (this.currentFrame >= this.totalFrames) {
        this.currentFrame = 0;
      }
      this.seekToFrame(this.currentFrame, false);
    }, intervalMs);
  }

  pause() {
    this.isPlaying = false;
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) playBtn.innerText = '▶';
    if (this.animationTimer) {
      clearInterval(this.animationTimer);
      this.animationTimer = null;
    }
  }

  seekToFrame(frameIdx, updateSlider = true) {
    this.currentFrame = Math.max(0, Math.min(frameIdx, this.totalFrames - 1));
    this.updateUI(updateSlider);

    for (let cb of this.callbacks) {
      cb(this.currentFrame);
    }
  }

  updateUI(updateSlider = true) {
    const slider = document.getElementById('timeline-scrubber');
    const readout = document.getElementById('player-clock-readout');
    const headerClock = document.getElementById('header-clock-text');

    if (slider && updateSlider) {
      slider.value = this.currentFrame;
    }

    const currentSec = this.startOffsetSec + (this.currentFrame / this.fps);
    const min = Math.floor(currentSec / 60);
    const sec = Math.floor(currentSec % 60);
    const clockStr = `${min}:${sec.toString().padStart(2, '0')}`;

    if (readout) readout.innerText = clockStr;
    if (headerClock) headerClock.innerText = `MATCH CLOCK: ${clockStr}`;
  }
}

window.gTimelinePlayer = new TimelinePlayer();
