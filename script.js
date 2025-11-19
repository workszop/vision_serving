
const VIDEO = document.getElementById('webcam');
const STATUS = document.getElementById('status');
const PREDS = document.getElementById('predictions');

let mobilenetModel;
let myModel;
let classes = [];

async function init() {
  try {
    // Load Metadata
    const metaRes = await fetch('metadata.json');
    const meta = await metaRes.json();
    classes = meta.classes;

    // Load Models
    STATUS.textContent = 'Loading MobileNet...';
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });

    STATUS.textContent = 'Loading Custom Model...';
    myModel = await tf.loadLayersModel('model.json');

    // Setup Camera
    STATUS.textContent = 'Starting Camera...';
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    VIDEO.srcObject = stream;
    await new Promise(r => VIDEO.onloadedmetadata = r);
    
    STATUS.textContent = 'Running...';
    predictLoop();
  } catch (err) {
    STATUS.textContent = 'Error: ' + err.message;
    console.error(err);
  }
}

async function predictLoop() {
  tf.tidy(() => {
    const img = tf.browser.fromPixels(VIDEO);
    const emb = mobilenetModel.infer(img, true);
    const prediction = myModel.predict(emb);
    const probs = prediction.dataSync();
    showPredictions(probs);
  });
  requestAnimationFrame(predictLoop);
}

function showPredictions(probs) {
  PREDS.innerHTML = '';
  let maxIdx = 0;
  for(let i=1; i<probs.length; i++) if(probs[i] > probs[maxIdx]) maxIdx = i;

  classes.forEach((cls, i) => {
    const p = probs[i];
    const pct = (p*100).toFixed(1);
    const isWinner = i === maxIdx;
    
    const div = document.createElement('div');
    div.className = `pred-item ${isWinner ? 'winner' : ''}`;
    div.innerHTML = `
      <div class="label">${cls}</div>
      <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="value">${pct}%</div>
    `;
    PREDS.appendChild(div);
  });
}

init();
