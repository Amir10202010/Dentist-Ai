// === help: small demo dataset ===
const demo = {
  patients: [
    { id:1, name:"Иванов Иван", phone:"+7 701 111 11 11", last:"2025-12-10", age:34, history:"Аллергии: нет" },
    { id:2, name:"Петрова Мария", phone:"+7 700 222 22 22", last:"2025-12-05", age:28, history:"Кариес" },
    { id:3, name:"Сидоров Алексей", phone:"+7 702 333 33 33", last:"2025-11-28", age:45, history:"Протезирование" }
  ],
  appointments: [
    { id:1, title:"Иванов — Чистка", start: new Date().toISOString().slice(0,10) + 'T10:00:00', end: new Date().toISOString().slice(0,10) + 'T10:30:00', doctor:"Dr. A" },
    { id:2, title:"Петрова — Консультация", start: new Date().toISOString().slice(0,10) + 'T12:00:00', end: new Date().toISOString().slice(0,10) + 'T12:30:00', doctor:"Dr. B" }
  ],
  revenue: { labels:["Пн","Вт","Ср","Чт","Пт","Сб","Вс"], data:[12000,15000,18000,9000,22000,10000,5000] },
  procedures: {labels:["Пломба","Отбеливание","Имплант"], data:[40,25,15]}
};

// ----------------- INIT -----------------
document.addEventListener("DOMContentLoaded", () => {
  initKPI();
  renderCharts();
  initPatients();
  initCalendar();
  initInteractive();
  bindTopbar();
});

// ----------------- Topbar minor bind -----------------
function bindTopbar(){
  const search = document.getElementById("searchInput");
  if(!search) return;
  search.addEventListener("input", (e)=>{
    const q = e.target.value.toLowerCase();
    filterPatients(q);
  });
}

// ----------------- KPI -----------------
function initKPI(){
  const totalPatients = demo.patients.length;
  const newToday = 1;
  const currentAppointments = demo.appointments.length;
  const future = demo.appointments.length;
  const missed = 0;
  const financeToday = 25000;
  const financeWeek = 89000;
  const completedPercent = 76;
  const avgDuration = 35;

  document.getElementById("kpi-patients-total").textContent = totalPatients;
  document.getElementById("kpi-patients-new").textContent = newToday;

  document.getElementById("kpi-appointments-current").textContent = currentAppointments;
  document.getElementById("kpi-appointments-future").textContent = future;
  document.getElementById("kpi-appointments-missed").textContent = missed;

  document.getElementById("kpi-finance-today").textContent = formatCurrency(financeToday);
  document.getElementById("kpi-finance-week").textContent = formatCurrency(financeWeek);

  document.getElementById("kpi-completed-percent").textContent = completedPercent + "%";
  document.getElementById("kpi-avg-duration").textContent = avgDuration + " мин";
}

// ----------------- Charts (Chart.js) -----------------
let chartRevenue, chartProcedures, chartPopular, chartByProcedure, chartDoctors;
function renderCharts(){
  const ctxR = document.getElementById("chartRevenue").getContext("2d");
  chartRevenue = new Chart(ctxR, {
    type: "line",
    data: { labels: demo.revenue.labels, datasets:[{ label:"Выручка", data: demo.revenue.data, tension:0.25, fill:true }]},
    options: { responsive:true, plugins:{legend:{display:false}}}
  });

  const ctxP = document.getElementById("chartProcedures").getContext("2d");
  chartProcedures = new Chart(ctxP, {
    type: "doughnut",
    data:{ labels: demo.procedures.labels, datasets:[{ data: demo.procedures.data }]},
    options:{ responsive:true }
  });

  chartPopular = new Chart(document.getElementById("chartPopular").getContext("2d"), {
    type:"bar",
    data:{ labels: ["Пломба","Отбеливание","Имплант"], datasets:[{label:"Пациенты", data:[40,25,15]}] },
    options:{ responsive:true, plugins:{legend:{display:false}}}
  });
  chartByProcedure = new Chart(document.getElementById("chartByProcedure").getContext("2d"), {
    type:"bar",
    data:{ labels: ["Пломба","Отбеливание","Имплант"], datasets:[{label:"Выручка", data:[400000,200000,600000]}] },
    options:{ responsive:true, plugins:{legend:{display:false}}}
  });
  chartDoctors = new Chart(document.getElementById("chartDoctors").getContext("2d"), {
    type:"bar",
    data:{ labels: ["Dr. A","Dr. B","Dr. C"], datasets:[{label:"Пациенты", data:[30,22,18]}] },
    options:{ responsive:true, plugins:{legend:{display:false}}}
  });
}

// ----------------- Patients -----------------
function initPatients(){
  renderPatients(demo.patients);

  document.getElementById("patientSearch")?.addEventListener("input", e=>{
    filterPatients(e.target.value);
  });

  document.getElementById("btnAddPatient")?.addEventListener("click", ()=>{
    showModal(true);
  });

  document.getElementById("btnClosePatient")?.addEventListener("click", ()=>showModal(false));
  document.getElementById("btnSavePatient")?.addEventListener("click", ()=>{
    const newP = {
      id: Date.now(),
      name: document.getElementById("p_name").value.trim(),
      phone: document.getElementById("p_phone").value.trim(),
      last: new Date().toISOString().slice(0,10),
      age: document.getElementById("p_age").value,
      history: document.getElementById("p_history").value
    };
    if(!newP.name || !newP.phone){
      alert("Заполните ФИО и телефон");
      return;
    }
    demo.patients.unshift(newP);
    renderPatients(demo.patients);
    showModal(false);
  });
}

function renderPatients(list){
  const tbody = document.querySelector("#patientsTable tbody");
  tbody.innerHTML = "";
  list.forEach(p=>{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(p.name)}</td>
      <td>${escapeHtml(p.phone)}</td>
      <td>${escapeHtml(p.last)}</td>
      <td>${escapeHtml(String(p.age||"—"))}</td>
      <td>
        <button class="table-action-btn" data-id="${p.id}" onclick="viewPatient(${p.id})">Открыть</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById("kpi-patients-total").textContent = demo.patients.length;
}

function filterPatients(q){
  const s = (q||"").toLowerCase().trim();
  const filtered = demo.patients.filter(p=>{
    return p.name.toLowerCase().includes(s) || (p.phone||"").toLowerCase().includes(s);
  });
  renderPatients(filtered);
}

function showModal(show){
  const m = document.getElementById("modalAddPatient");
  if(!m) return;
  m.style.display = show ? "flex" : "none";
  if(show){
    document.getElementById("p_name").value = "";
    document.getElementById("p_phone").value = "";
    document.getElementById("p_age").value = "";
    document.getElementById("p_history").value = "";
  }
}

function viewPatient(id){
  const p = demo.patients.find(x=>x.id===id);
  if(!p) return alert("Пациент не найден");
  alert(`Пациент: ${p.name}\nТел: ${p.phone}\nПоследний визит: ${p.last}\nИстория: ${p.history}`);
}

// ----------------- Calendar -----------------
function initCalendar(){
  const calendarEl = document.getElementById("calendar");
  if(!calendarEl) return;

  const calendar = new FullCalendar.Calendar(calendarEl, {
    initialView: 'dayGridMonth',
    height: 520,
    editable: true,
    selectable: true,
    headerToolbar: { left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek,timeGridDay' },
    events: demo.appointments,
    eventDrop: info => console.log('Event dropped to', info.event.start),
    select: info => {
      const title = prompt("Название записи (например: Иванов — Чистка)");
      if(title) calendar.addEvent({ title, start: info.startStr, end: info.endStr, allDay: info.allDay });
      calendar.unselect();
    },
    eventClick: info => { if(confirm("Удалить запись?")) info.event.remove(); }
  });

  calendar.render();
}

// ----------------- Interactive -----------------
function initInteractive(){
  document.getElementById("fileInputMain")?.addEventListener("change", async e=>{
    const f = e.target.files[0];
    if(!f) return;
    document.getElementById("uploadStatus").textContent = `Загружен: ${f.name} — анализ... (демо)`;
    setTimeout(()=> document.getElementById("uploadStatus").textContent = `Анализ завершён (демо)` , 1400);
    e.target.value = "";
    const el = document.getElementById("totalUploads");
    if(el) el.textContent = Number(el.textContent||0)+1;
  });

  document.getElementById("chatSend")?.addEventListener("click", ()=>{
    const txt = document.getElementById("chatInput").value.trim();
    if(!txt) return;
    const box = document.getElementById("chatBox");
    const p = document.createElement("div"); p.textContent = "Вы: " + txt; p.style.marginBottom="6px";
    box.appendChild(p);
    document.getElementById("chatInput").value = "";
    box.scrollTop = box.scrollHeight;
  });

  const remList = document.getElementById("remindersList");
  ["Напомнить Иванову о приёме — 2025-12-14","Просрочен приём — Петрова"].forEach(r=>{
    const li=document.createElement("li"); li.textContent = r; remList.appendChild(li);
  });
}

// ----------------- Utils -----------------
function formatCurrency(n){ return '₸' + Number(n).toLocaleString(); }
function escapeHtml(s=""){ return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }
