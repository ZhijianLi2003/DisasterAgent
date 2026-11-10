'use strict';
const config = window.DISASTERAGENT_CONFIG || {};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let toastTimer;
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 4200);
}

const placeholder = (url) => !url || /XXXX|your-org|your-username/.test(url);
$$('[data-resource]').forEach(link => {
  const url = config[link.dataset.resource === 'paper' ? 'paperUrl' : 'codeUrl'];
  if (url) link.href = url;
  if (placeholder(url)) {
    link.addEventListener('click', event => {
      event.preventDefault();
      toast(link.dataset.resource === 'paper' ? 'The arXiv link will be added with the public release.' : 'The GitHub repository link will be added with the public release.');
    });
  }
});
if (config.arxivId) {
  $('#bibtex').textContent = $('#bibtex').textContent.replace('journal = {arXiv preprint},', `journal = {arXiv preprint arXiv:${config.arxivId}},\n  eprint  = {${config.arxivId}},\n  archivePrefix = {arXiv},`);
  $('#citation-note').hidden = true;
}
if (config.year) $('#bibtex').textContent = $('#bibtex').textContent.replace('li2026disasteragent', `li${config.year}disasteragent`).replace('year    = {2026}', `year    = {${config.year}}`);

const video = $('#demo-video');
function updateSound() {
  const audible = !video.muted && video.volume > 0;
  $('#sound-toggle').setAttribute('aria-pressed', String(audible));
  $('#sound-label').textContent = audible ? 'Mute music' : 'Enable music';
}
$('#sound-toggle').addEventListener('click', () => {
  video.muted = !video.muted;
  if (!video.muted && video.volume === 0) video.volume = 1;
  updateSound();
});
video.addEventListener('volumechange', updateSound);
$$('[data-time]').forEach(button => button.addEventListener('click', () => {
  video.currentTime = Number(button.dataset.time);
  video.play().catch(() => toast('Press play in the video player to continue.'));
}));
video.addEventListener('timeupdate', () => {
  const buttons = $$('[data-time]');
  buttons.forEach((button, index) => {
    const active = video.currentTime >= Number(button.dataset.time) && (!buttons[index + 1] || video.currentTime < Number(buttons[index + 1].dataset.time));
    if (active) button.setAttribute('aria-current', 'true'); else button.removeAttribute('aria-current');
  });
});

function activateTab(button) {
  const group = button.closest('[role="tablist"]');
  group.querySelectorAll('[role="tab"]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab === button));
    tab.tabIndex = tab === button ? 0 : -1;
  });
  document.getElementById(button.getAttribute('aria-controls')).setAttribute('aria-labelledby', button.id);
}
$$('[role="tablist"]').forEach(group => group.addEventListener('keydown', event => {
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const current = tabs.indexOf(document.activeElement);
  if (current < 0 || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
  event.preventDefault();
  let next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  tabs[next].click();
}));

const cases = {
  explosion: ['01','Explosion','From a misidentified event to an evidence-grounded explanation: DisasterAgent links the explosion assessment to numbered visual references.',1885],
  tsunami: ['02','Tsunami','Compare the image pair, generated descriptions, and annotated evidence in a tsunami-affected scene.',2088],
  earthquake: ['03','Earthquake','Inspect the earthquake example to see how the report connects observed damage to spatially grounded references.',1885],
  flood: ['04','Flood','Follow the flood assessment from paired observations to a disaster description supported by numbered visual evidence.',2088],
  wildfire: ['05','Wildfire','Explore the wildfire example and the visual references that support the generated account of disaster-induced changes.',2009]
};
$$('[data-case]').forEach(button => button.addEventListener('click', () => {
  activateTab(button);
  const key = button.dataset.case;
  const [number,title,description,height] = cases[key];
  $('#case-number').textContent = `CASE ${number}`;
  $('#case-title').textContent = title;
  $('#case-description').textContent = description;
  const img = $('#case-image');
  img.src = `assets/case-${key}.webp`; img.width = 2400; img.height = height;
  img.alt = `${title} case: input image pair, baseline output, DisasterAgent report, and annotated visual evidence`;
  $('#case-zoom').dataset.zoom = img.getAttribute('src');
  $('#case-zoom').dataset.caption = `${title}: qualitative comparison from the DisasterAgent paper.`;
}));

const benchmarks = {
  m3: {
    title:'Disaster report generation on DisasterM3', description:'2,512 optical image pairs · disaster report generation',
    groups:[{label:'Disaster captioning',span:4},{label:'Restoration advice',span:4}],
    headers:['Method','AVG','DAP','DDR','FC','AVG','RN','APP','SC'],
    headerDetails:['Method','Average','Damage Assessment Precision','Damage Detail Recall','Factual Correctness','Average','Recovery Necessity','Action Priority Precision','Strategic Completeness'],
    directions:[1,1,1,1,1,1,1,1],
    rows:[
      ['Kimi K2.6','1.37','1.61','1.21','1.30','2.96','3.52','2.78','2.57'],
      ['MiniMax-M3','2.03','2.31','1.80','1.97','2.79','3.37','2.58','2.42'],
      ['Qwen3.5-27B','1.88','2.16','1.65','1.81','2.86','3.47','2.67','2.44'],
      ['Qwen3.5-9B','1.71','2.02','1.47','1.65','2.62','3.23','2.43','2.19'],
      ['InternVL3.5-8B','1.16','1.43','0.97','1.09','2.55','3.26','2.26','2.14'],
      ['GeoChat-7B','0.78','0.97','0.34','1.03','1.72','2.18','1.60','1.39'],
      ['TeoChat-7B','1.29','1.56','0.64','1.65','1.62','2.11','1.47','1.29'],
      ['EarthAgent','2.36','2.72','2.03','2.34','3.15','3.76','2.96','2.74'],
      ['OpenEarthAgent','2.28','2.61','1.94','2.29','3.13','3.76','2.92','2.70'],
      ['DisasterAgent','2.87','3.20','2.52','2.88','3.22','3.77','3.05','2.84']
    ]
  },
  rscc: {
    title:'Performance on RSCC', description:'988 image pairs · change captioning',
    headers:['Method','ROUGE-L\n(%) ↑','METEOR\n(%) ↑','ST5-SCS\n(%) ↑'],directions:[1,1,1],
    rows:[['Kimi K2.6','11.14','10.98','52.29'],['GLM-4.5V','13.24','10.64','50.57'],['InternVL3.5-8B','12.05','10.84','47.98'],['Qwen3.5-27B','11.68','11.00','51.84'],['Qwen3.5-9B','13.23','11.23','49.25'],['CCExpert-7B','8.39','3.73','40.10'],['RSCCM-7B','14.99','16.05','58.52'],['EarthAgent','15.31','13.50','65.88'],['OpenEarthAgent','16.00','12.74','62.21'],['DisasterAgent','15.76','14.77','67.18']]
  }
};
function renderBenchmark(key) {
  const data = benchmarks[key];
  $('#benchmark-description').textContent = data.description;
  const table = $('#benchmark-table'); table.replaceChildren();
  table.classList.toggle('report-table', Boolean(data.groups));
  const caption = table.createCaption(); caption.className = 'sr-only'; caption.textContent = data.title;
  const head = table.createTHead();
  if (data.groups) {
    const groupRow = head.insertRow();
    const method = document.createElement('th');
    method.scope = 'col'; method.rowSpan = 2; method.textContent = data.headers[0]; groupRow.append(method);
    data.groups.forEach(group => {
      const th = document.createElement('th'); th.scope = 'colgroup'; th.colSpan = group.span;
      th.textContent = group.label;
      const scale = document.createElement('span'); scale.className = 'score-scale'; scale.textContent = 'Scores / 5 ↑';
      th.append(scale); groupRow.append(th);
    });
    const metricRow = head.insertRow();
    data.headers.slice(1).forEach((label,index) => {
      const th = document.createElement('th'); th.scope = 'col';
      const abbr = document.createElement('abbr'); abbr.textContent = label; abbr.title = data.headerDetails[index + 1];
      th.append(abbr); metricRow.append(th);
    });
  } else {
    const row = head.insertRow();
    data.headers.forEach(label => {const th = document.createElement('th'); th.scope = 'col'; th.style.whiteSpace = 'pre-line'; th.textContent = label; row.append(th);});
  }
  const best = data.directions.map((direction,index) => Math[direction > 0 ? 'max' : 'min'](...data.rows.map(row => Number(row[index+1]))));
  const body = table.createTBody();
  data.rows.forEach(row => {
    const tr = body.insertRow();
    if (row[0].startsWith('DisasterAgent')) tr.className = 'ours';
    row.forEach((value,index) => {
      const cell = document.createElement(index === 0 ? 'th' : 'td');
      if (index === 0) cell.scope = 'row';
      else if (Number(value) === best[index-1]) cell.classList.add('best');
      cell.textContent = value; tr.append(cell);
    });
  });
}
$$('[data-benchmark]').forEach(button => button.addEventListener('click', () => {activateTab(button); renderBenchmark(button.dataset.benchmark);}));
renderBenchmark('m3');

const dialog = $('#figure-dialog');
let previousFocus;
$$('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  previousFocus = button;
  $('#dialog-image').src = button.dataset.zoom;
  $('#dialog-image').alt = button.querySelector('img').alt;
  $('#dialog-caption').textContent = button.dataset.caption;
  $('#dialog-image-area').classList.remove('original');
  $('#zoom-toggle').textContent = 'Original size';
  $('#zoom-toggle').setAttribute('aria-pressed','false');
  dialog.showModal();
  document.body.style.overflow = 'hidden';
}));
$('#dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {if (event.target === dialog) {const r = dialog.getBoundingClientRect(); if(event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();}});
dialog.addEventListener('close', () => {document.body.style.overflow = ''; previousFocus?.focus({preventScroll:true});});
$('#zoom-toggle').addEventListener('click', () => {
  const original = $('#dialog-image-area').classList.toggle('original');
  $('#zoom-toggle').textContent = original ? 'Fit to screen' : 'Original size';
  $('#zoom-toggle').setAttribute('aria-pressed', String(original));
});
$('#copy-citation').addEventListener('click', async () => {
  const value = $('#bibtex').textContent;
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(value);
    toast('Citation copied to clipboard.');
    $('#copy-citation').textContent = 'Copied ✓';
    setTimeout(() => $('#copy-citation').textContent = 'Copy citation ⧉', 2200);
  } catch {
    const range = document.createRange(); range.selectNodeContents($('#bibtex'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    toast('Citation selected. Press Ctrl+C (or ⌘C) to copy.');
  }
});

const navLinks = $$('.nav-wrap nav a');
const sections = navLinks.map(link => document.querySelector(link.getAttribute('href')));
let scrollPending = false;
function updateScroll() {
  const total = document.documentElement.scrollHeight - innerHeight;
  $('#reading-progress').style.transform = `scaleX(${total > 0 ? Math.max(0,Math.min(1,scrollY / total)) : 0})`;
  let active = '';
  sections.forEach(section => {if (section.getBoundingClientRect().top < 190) active = `#${section.id}`;});
  navLinks.forEach(link => {if(link.getAttribute('href') === active) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current');});
  scrollPending = false;
}
addEventListener('scroll', () => {if (!scrollPending) {scrollPending = true; requestAnimationFrame(updateScroll);}}, {passive:true});
addEventListener('resize', updateScroll);
updateScroll();
