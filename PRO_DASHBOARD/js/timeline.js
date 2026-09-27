/**
 * Interactive Timeline & Animation Player Engine
 * Controls match clock scrubber (9:54 to 24:54), playback speeds (0.5x, 1x, 2x, 5x),
 * and syncs frame state across pitch canvas, graphs, and metric cards.
 */
class ProTimelinePlayer {
  constructor() {
    this.isPlaying = false;
    this.currentFrame = 0;
    this.totalFrames = 22500;
    this.fps = 25.0;
    this.clipStartOffsetSec = 594.0; // 9:54 match clock start
    this.playbackSpeed = 1.0;
    this.timerId = null;

    this.onFrameCallbacks = [];
  }

  init(totalFrames, fps = 25.0, startOffsetSec = 594.0) {
    this.totalFrames = totalFrames;
    this.fps = fps;
    this.clipStartOffsetSec = startOffsetSec;

    this.btnPlay = document.getElementById('btn-play-pause');
    this.scrubber = document.getElementById('timeline-scrubber');
    this.clockReadout = document.getElementById('player-clock-readout');

    if (this.scrubber) {
      this.scrubber.max = this.totalFrames - 1;
      this.scrubber.value = 0;
      this.scrubber.addEventListener('input', (e) => {
        this.seek(parseInt(e.target.value));
      });
    }

    if (this.btnPlay) {
      this.btnPlay.addEventListener('click', () => this.togglePlay());
    }

    // Speed selector buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.setSpeed(parseFloat(e.target.getAttribute('data-speed')));
      });
    });

    this.updateUI();
  }

  addFrameCallback(fn) {
    this.onFrameCallbacks.push(fn);
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    if (this.btnPlay) this.btnPlay.innerHTML = '❚❚';

    const intervalMs = (1000 / this.fps) / this.playbackSpeed;
    this.timerId = setInterval(() => {
      this.currentFrame += 5; // Step by 5 frames for smooth visualization
      if (this.currentFrame >= this.totalFrames) {
        this.currentFrame = 0;
      }
      this.notifyFrame();
      this.updateUI();
    }, intervalMs);
  }

  pause() {
    if (!this.isPlaying) return;
    this.isPlaying = false;
    if (this.btnPlay) this.btnPlay.innerHTML = '▶';
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  setSpeed(speed) {
    this.playbackSpeed = speed;
    if (this.isPlaying) {
      this.pause();
      this.play();
    }
  }

  seek(frameIdx) {
    this.currentFrame = Math.max(0, Math.min(this.totalFrames - 1, frameIdx));
    this.notifyFrame();
    this.updateUI();
  }

  notifyFrame() {
    this.onFrameCallbacks.forEach(cb => cb(this.currentFrame));
  }

  updateUI() {
    if (this.scrubber) {
      this.scrubber.value = this.currentFrame;
    }

    const totalSec = this.clipStartOffsetSec + (this.currentFrame / this.fps);
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    const timeStr = `${min}:${sec.toString().padStart(2, '0')}`;

    if (this.clockReadout) {
      this.clockReadout.innerText = timeStr;
    }
  }
}
