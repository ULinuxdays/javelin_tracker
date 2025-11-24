const state = {
  group: 'week',
  summary: null,
  sessions: [],
  chart: null,
  summaryTab: 'all',
  timelineFilter: 'all',
  searchQuery: '',
  athletes: [],
  plan: [],
  forecast: null,
  strengthLogs: [],
  strengthFilter: 'all',
};

const page = document.body.dataset.page || null;
const controllers = {
  dashboard: initDashboard,
  sessions: initSessions,
  logs: initLogs,
  analytics: initAnalytics,
  athletes: initAthletes,
  throwai: initTraining,
  weightroom: initWeightRoom,
  reports: initReports,
};

async function ensureAthletes(force = false) {
  if (!force && state.athletes.length) {
    populateAthleteFields();
    return state.athletes;
  }
  try {
    const response = await fetch('/api/athletes');
    const data = await response.json();
    state.athletes = data.athletes || [];
    populateAthleteFields();
    return state.athletes;
  } catch (error) {
    console.error(error);
    showToast('Unable to load athletes.');
    return [];
  }
}

function populateAthleteFields() {
  const selectOptions =
    '<option value="">Select athlete</option>' +
    state.athletes.map((athlete) => `<option value="${athlete.name}">${athlete.name}</option>`).join('');
  document.querySelectorAll('[data-athlete-select]').forEach((element) => {
    element.innerHTML = selectOptions;
    element.disabled = state.athletes.length === 0;
  });
  const datalist = document.getElementById('athleteOptions');
  if (datalist) {
    datalist.innerHTML = state.athletes
      .map((athlete) => `<option value="${athlete.name}"></option>`)
      .join('');
  }
}

function renderAthletesTable() {
  const tbody = document.getElementById('athletesTableBody');
  if (!tbody) {
    return;
  }
  if (!state.athletes.length) {
    tbody.innerHTML = '<tr><td colspan="6">No athletes in the database yet.</td></tr>';
    return;
  }
  tbody.innerHTML = state.athletes
    .map(
      (athlete) => `
        <tr data-athlete-id="${athlete.id}">
          <td>${athlete.name}</td>
          <td>${athlete.height_cm ?? '—'}</td>
          <td>${athlete.weight_kg ?? '—'}</td>
          <td>${athlete.bmi != null ? formatNumber(athlete.bmi, 1) : '—'}</td>
          <td>${athlete.bench_1rm_kg != null ? formatNumber(athlete.bench_1rm_kg, 1) + ' kg' : '—'}</td>
          <td>${athlete.squat_1rm_kg != null ? formatNumber(athlete.squat_1rm_kg, 1) + ' kg' : '—'}</td>
        </tr>
      `
    )
    .join('');

  tbody.querySelectorAll('tr[data-athlete-id]').forEach((row) => {
    row.addEventListener('click', () => {
      const id = Number(row.dataset.athleteId);
      const athlete = state.athletes.find((a) => a.id === id);
      if (athlete) {
        fillAthleteForm(athlete);
      }
    });
  });

  const selectField = document.getElementById('athleteSelect');
  if (selectField) {
    const previous = selectField.value;
    selectField.innerHTML =
      '<option value="">Select athlete</option>' +
      state.athletes.map((athlete) => `<option value="${athlete.id}">${athlete.name}</option>`).join('');
    let selected = state.athletes.find((a) => String(a.id) === previous);
    if (!selected && state.athletes.length) {
      selected = state.athletes[0];
    }
    if (selected) {
      selectField.value = String(selected.id);
      fillAthleteForm(selected);
    }
  } else if (state.athletes[0]) {
    fillAthleteForm(state.athletes[0]);
  }
}

function fillAthleteForm(athlete) {
  const idField = document.getElementById('athleteId');
  const selectField = document.getElementById('athleteSelect');
  const heightField = document.getElementById('athleteHeight');
  const weightField = document.getElementById('athleteWeight');
  const bmiField = document.getElementById('athleteBmi');
  const benchField = document.getElementById('athleteBench');
  const squatField = document.getElementById('athleteSquat');
  const notesField = document.getElementById('athleteNotes');
  if (!idField) {
    return;
  }
  idField.value = athlete.id;
  if (selectField) {
    selectField.value = String(athlete.id);
  }
  if (heightField) heightField.value = athlete.height_cm ?? '';
  if (weightField) weightField.value = athlete.weight_kg ?? '';
  if (bmiField) bmiField.value = athlete.bmi != null ? formatNumber(athlete.bmi, 1) : '';
  if (benchField) benchField.value = athlete.bench_1rm_kg ?? '';
  if (squatField) squatField.value = athlete.squat_1rm_kg ?? '';
  if (notesField) notesField.value = athlete.notes ?? '';
}

async function saveAthleteProfile(formData) {
  const id = formData.get('id');
  if (!id) {
    showToast('Select an athlete first.');
    return;
  }
  const payload = Object.fromEntries(formData.entries());
  try {
    const response = await fetch(`/api/athletes/${id}/profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to save profile.');
    }
    const updated = data.athlete;
    // Update state
    const idx = state.athletes.findIndex((a) => a.id === updated.id);
    if (idx >= 0) {
      state.athletes[idx] = updated;
    } else {
      state.athletes.push(updated);
    }
    renderAthletesTable();
    fillAthleteForm(updated);
    populateAthleteFields();
    showToast('Athlete profile updated.');
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Unable to save profile.');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  if (!page || !(page in controllers)) {
    return;
  }
  bindCommonInteractions();
  controllers[page]();
});

function bindCommonInteractions() {
  document.querySelectorAll('[data-summary-toggle]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (button.dataset.summaryToggle === state.group) {
        return;
      }
      document
        .querySelectorAll('[data-summary-toggle]')
        .forEach((el) => el.classList.toggle('active', el === button));
      state.group = button.dataset.summaryToggle;
      await loadSummary();
    });
  });

  document.querySelectorAll('[data-tab]').forEach((button) => {
    button.addEventListener('click', () => {
      state.summaryTab = button.dataset.tab;
      document
        .querySelectorAll('[data-tab]')
        .forEach((el) => el.classList.toggle('active', el === button));
      renderSummaryTable();
    });
  });

  const filter = document.getElementById('timelineFilter');
  if (filter) {
    filter.addEventListener('change', () => {
      state.timelineFilter = filter.value;
      renderTimeline();
      renderSessionsTable();
    });
  }

  const search = document.getElementById('sessionsSearch');
  if (search) {
    search.addEventListener('input', () => {
      state.searchQuery = search.value.trim().toLowerCase();
      renderSessionsTable();
    });
  }

  const strengthFilter = document.getElementById('strengthFilter');
  if (strengthFilter) {
    strengthFilter.addEventListener('change', () => {
      state.strengthFilter = strengthFilter.value;
      renderStrengthLogs();
    });
  }

  const form = document.getElementById('sessionForm');
  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(buildFormDataWithAthlete(form).entries());
      if (!payload.event) {
        showToast('Event is required to log a session.');
        return;
      }
      setFormStatus('Saving...');
      try {
        const response = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Unable to save session.');
        }
        form.reset();
        showToast('Session saved and analytics refreshed.');
        await Promise.all([loadSummary(), loadSessions()]);
      } catch (error) {
        console.error(error);
        showToast(error.message || 'An unexpected error occurred.');
      } finally {
        setFormStatus('Ready');
      }
    });
  }

  document.querySelectorAll('[data-scroll-target]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = document.querySelector(button.dataset.scrollTarget);
      target?.scrollIntoView({ behavior: 'smooth' });
    });
  });

  document.querySelectorAll('.floating-card').forEach((card) => {
    card.addEventListener('mousemove', (event) => {
      const rect = card.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      card.style.transform = `translateY(-4px) rotateX(${(0.5 - y) * 6}deg) rotateY(${(x - 0.5) * 6}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  const cliButton = document.getElementById('refreshCliSummary');
  if (cliButton) {
    cliButton.addEventListener('click', () => refreshCliSummary());
  }

  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-delete-session]');
    if (target) {
      deleteSession(target.dataset.deleteSession);
    }
    const toggle = event.target.closest('[data-toggle-strength]');
    if (toggle) {
      toggleStrengthGroup(toggle.dataset.toggleStrength, toggle);
    }
    const logButton = event.target.closest('[data-plan-log]');
    if (logButton) {
      logPlanRow(logButton.dataset.planLog);
    }
  });

  document.addEventListener('change', (event) => {
    const select = event.target.closest('[data-logged-athlete-select]');
    if (select) {
      toggleLoggedAthleteInput(select);
    }
  });

  document.querySelectorAll('[data-nav]').forEach((element) => {
    element.addEventListener('click', () => {
      const target = element.dataset.nav;
      if (target) {
        window.location.href = target;
      }
    });
  });

  const dailyPlanForm = document.getElementById('dailyPlanForm');
  if (dailyPlanForm) {
    dailyPlanForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      await requestDailyPlan(new FormData(dailyPlanForm));
    });
  }

  document.querySelectorAll('[data-lift-form]').forEach((form) => {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      await submitLift(buildFormDataWithAthlete(form), { statusId: form.dataset.statusTarget, form });
    });
  });

  const forecastForm = document.getElementById('forecastForm');
  if (forecastForm) {
    forecastForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      await requestForecast(new FormData(forecastForm));
    });
  }
}

async function initDashboard() {
  await Promise.all([loadSummary(), loadSessions()]);
}

async function initSessions() {
  await Promise.all([loadSessions(), ensureAthletes()]);
}

async function initLogs() {
  await Promise.all([loadSessions(), loadStrengthLogs(), ensureAthletes()]);
  renderStrengthLogs();
}

async function initAnalytics() {
  await Promise.all([loadSummary(), loadSessions(), ensureAthletes()]);
  await refreshCliSummary();
}

async function initTraining() {
  await Promise.all([loadSummary(), loadSessions(), ensureAthletes()]);
}

async function initReports() {
  await Promise.all([loadSummary(), refreshCliSummary(), ensureAthletes()]);
  const form = document.getElementById('weeklyReportForm');
  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      await requestWeeklyReport(new FormData(form));
    });
  }
}

async function initWeightRoom() {
  await Promise.all([ensureAthletes(), loadSessions()]);
}

async function initAthletes() {
  try {
    const response = await fetch('/api/athletes/profiles');
    const data = await response.json();
    state.athletes = data.athletes || [];
    renderAthletesTable();
    populateLoggedAthleteSelects();
  } catch (error) {
    console.error(error);
    showToast('Unable to load athlete profiles.');
  }
  const form = document.getElementById('athleteForm');
  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      await saveAthleteProfile(new FormData(form));
    });
  }
  const selectField = document.getElementById('athleteSelect');
  if (selectField) {
    selectField.addEventListener('change', () => {
      const id = Number(selectField.value);
      const athlete = state.athletes.find((a) => a.id === id);
      if (athlete) {
        fillAthleteForm(athlete);
      }
    });
  }
}

async function loadSummary() {
  try {
    const response = await fetch(`/api/summary?group=${state.group}`);
    const data = await response.json();
    state.summary = data;
    renderStats();
    renderChart();
    renderSummaryTable();
    renderPersonalBests();
    renderRiskList();
    renderTrainingIntel();
  } catch (error) {
    console.error(error);
    showToast('Unable to load summary.');
  }
}

async function loadSessions() {
  try {
    const response = await fetch('/api/sessions');
    const data = await response.json();
    state.sessions = data.sessions || [];
    populateTimelineFilter();
    renderTimeline();
    renderSessionsTable();
    updateHeroHighlights();
    renderTrainingTimeline();
    renderTagFocus();
    populateLoggedAthleteSelects();
  } catch (error) {
    console.error(error);
    showToast('Unable to load sessions.');
  }
}

async function loadStrengthLogs() {
  try {
    const response = await fetch('/api/strength-logs');
    const data = await response.json();
    state.strengthLogs = data.logs || [];
    populateStrengthFilter();
    renderStrengthLogs();
  } catch (error) {
    console.error(error);
    showToast('Unable to load workout logs.');
  }
}

function renderStats() {
  if (!state.summary) {
    return;
  }
  const totals = state.summary.summary?.totals ?? {};
  const statSessions = document.getElementById('statSessions');
  if (statSessions) {
    statSessions.textContent = totals.sessions ?? 0;
  }
  const statThrows = document.getElementById('statThrows');
  if (statThrows) {
    statThrows.textContent = totals.throws ?? 0;
  }
  const statLoad = document.getElementById('statLoad');
  if (statLoad) {
    statLoad.textContent = `${formatNumber(totals.load, 1)} AU`;
  }
  const avgRpe = totals.averageRpe != null ? formatNumber(totals.averageRpe, 1) : '--';
  const statAvgRpe = document.getElementById('statAvgRpe');
  if (statAvgRpe) {
    statAvgRpe.textContent = `Avg RPE ${avgRpe}`;
  }
  const statDateRange = document.getElementById('statDateRange');
  if (statDateRange && totals.dateRange) {
    statDateRange.textContent = `${totals.dateRange.start} → ${totals.dateRange.end}`;
  }
  const pb =
    state.summary.personalBest ||
    state.summary.personal_best ||
    state.summary.summary?.personalBest ||
    null;
  const statPb = document.getElementById('statPb');
  if (statPb) {
    statPb.textContent = pb ? `${formatNumber(pb.best)} m` : '--';
  }
  const statPbDate = document.getElementById('statPbDate');
  if (statPbDate) {
    statPbDate.textContent = pb ? `Set ${pb.date}` : 'Log a session to set a PB';
  }
  const heroBest = document.getElementById('heroBest');
  if (heroBest) {
    heroBest.textContent = pb ? `${formatNumber(pb.best)} m` : '--';
  }
  const heroPB = document.getElementById('heroPersonalBest');
  if (heroPB) {
    heroPB.textContent = pb ? `${formatNumber(pb.best)} m` : '--';
  }
  const heroPBDate = document.getElementById('heroPersonalBestDate');
  if (heroPBDate) {
    heroPBDate.textContent = pb ? pb.date : '--';
  }
  const heroSessions = document.getElementById('heroSessions');
  if (heroSessions) {
    heroSessions.textContent = totals.sessions ?? 0;
  }
  const heroLoad = document.getElementById('heroLoad');
  if (heroLoad) {
    heroLoad.textContent = `${formatNumber(totals.load, 1)} AU`;
  }
  renderReadinessSummary();
}

function renderChart() {
  const canvas = document.getElementById('performanceChart');
  if (!state.summary || !canvas) {
    toggleChartPlaceholder(true);
    return;
  }
  const series = state.summary.series ?? [];
  if (!series.length) {
    if (state.chart) {
      state.chart.destroy();
      state.chart = null;
    }
    toggleChartPlaceholder(true);
    return;
  }
  toggleChartPlaceholder(false);
  const labels = series.map((item) => item.label);
  const mean = series.map((item) => item.meanBest);
  const median = series.map((item) => item.medianBest);
  const volume = series.map((item) => item.throws);
  if (state.chart) {
    state.chart.destroy();
  }
  state.chart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Mean best',
          data: mean,
          borderColor: '#64f2ff',
          backgroundColor: 'rgba(100, 242, 255, 0.2)',
          tension: 0.4,
          borderWidth: 2,
        },
        {
          label: 'Median best',
          data: median,
          borderColor: '#ff7ed4',
          backgroundColor: 'rgba(255, 126, 212, 0.15)',
          tension: 0.4,
          borderDash: [6, 6],
          borderWidth: 2,
        },
        {
          label: 'Throw volume',
          data: volume,
          type: 'bar',
          backgroundColor: 'rgba(109, 229, 154, 0.25)',
          borderColor: '#6de59a',
          yAxisID: 'volume',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#aab2cc' } },
        volume: { beginAtZero: true, position: 'right', grid: { display: false }, ticks: { color: '#aab2cc' } },
        x: { grid: { display: false }, ticks: { color: '#aab2cc' } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(5,7,13,0.9)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
        },
      },
    },
  });
}

function renderSummaryTable() {
  const table = document.getElementById('summaryTable');
  if (!state.summary || !table) {
    return;
  }
  const rows =
    state.summaryTab === 'all'
      ? state.summary.summary?.rollupRows
      : state.summary.summary?.rows;
  if (!rows || rows.length === 0) {
    table.innerHTML = `<tr><td colspan="10">No sessions logged yet.</td></tr>`;
    const heroAlerts = document.getElementById('heroAlerts');
    if (heroAlerts) {
      heroAlerts.textContent = '0';
    }
    return;
  }
  const formatter = (value, suffix = '') => {
    const formatted = formatNumber(value, 1);
    return formatted === '—' ? '—' : `${formatted}${suffix}`;
  };
  table.innerHTML = rows
    .map((row) => {
      const riskBadge = row.risk ? `<span class="tag">${row.risk}</span>` : '—';
      return `
        <tr>
          <td>${row.scope.toUpperCase()}</td>
          <td>${row.athlete || '—'}</td>
          <td>${row.event || '—'}</td>
          <td>${row.label}</td>
          <td>${row.sessions}</td>
          <td>${formatter(row.best, ' m')}</td>
          <td>${formatter(row.load, ' AU')}</td>
          <td>${row.throws}</td>
          <td>${formatter(row.acwrRolling)}</td>
          <td>${riskBadge}</td>
        </tr>
      `;
    })
    .join('');
  const heroAlerts = document.getElementById('heroAlerts');
  if (heroAlerts) {
    heroAlerts.textContent = rows.filter((row) => row.risk).length;
  }
}

function renderPersonalBests(targetId = 'pbList') {
  const list = document.getElementById(targetId);
  if (!state.summary || !list) {
    return;
  }
  const entries = state.summary.summary?.personalBests || [];
  if (entries.length === 0) {
    list.innerHTML = '<li class="intel-card">No PBs logged yet.</li>';
    const summary = document.getElementById('intelPbSummary');
    if (summary) {
      summary.textContent = 'Log sessions to start tracking PBs.';
    }
    return;
  }
  list.innerHTML = entries
    .map(
      (entry) => `
        <li class="intel-card">
          <div class="intel-primary">${formatNumber(entry.best)} m</div>
          <div class="intel-meta">${entry.athlete} · ${entry.event}</div>
          <small>${entry.date}</small>
        </li>
      `
    )
    .join('');
  const summary = document.getElementById('intelPbSummary');
  if (summary) {
    const headline = entries[0];
    summary.textContent = `Tracking ${entries.length} PB groups; latest: ${headline.athlete} (${headline.event}) at ${formatNumber(
      headline.best
    )} m.`;
  }
}

function renderRiskList(targetId = 'riskList') {
  const list = document.getElementById(targetId);
  if (!state.summary || !list) {
    return;
  }
  const entries = state.summary.summary?.highRiskDates || {};
  const items = Object.entries(entries);
  if (items.length === 0) {
    list.innerHTML = '<li class="intel-card">All clear — keep building momentum.</li>';
    const summary = document.getElementById('intelRiskSummary');
    if (summary) {
      summary.textContent = 'No high-risk windows flagged.';
    }
    return;
  }
  list.innerHTML = items
    .map(([key, dates]) => {
      const [athlete, event] = key.split('|');
      const chips = dates.map((date) => `<span class="intel-chip">${date}</span>`).join('');
      return `
        <li class="intel-card">
          <div class="intel-meta">${athlete} · ${event}</div>
          <div class="intel-chips">${chips}</div>
        </li>
      `;
    })
    .join('');
  const summary = document.getElementById('intelRiskSummary');
  if (summary) {
    summary.textContent = `${items.length} athlete/event group${items.length === 1 ? '' : 's'} require attention.`;
  }
}

function populateTimelineFilter() {
  const filter = document.getElementById('timelineFilter');
  if (!filter) {
    return;
  }
  const current = filter.value;
  const events = Array.from(
    new Set(state.sessions.map((session) => (session.event || '').toLowerCase()).filter(Boolean))
  );
  const options = ['all', ...events];
  filter.innerHTML = options
    .map((value) => `<option value="${value}">${value === 'all' ? 'All' : value}</option>`)
    .join('');
  filter.value = options.includes(current) ? current : 'all';
  state.timelineFilter = filter.value;
}

function populateLoggedAthleteSelects() {
  const sessionNames = state.sessions.map((session) => (session.athlete || '').trim()).filter(Boolean);
  const profileNames = (state.athletes || []).map((athlete) => athlete.name || '').filter(Boolean);
  const names = Array.from(new Set([...sessionNames, ...profileNames])).sort((a, b) => a.localeCompare(b));
  const options = ['<option value="">Select athlete</option>'];
  names.forEach((name) => {
    options.push(`<option value="${name}">${name}</option>`);
  });
  options.push('<option value="__custom">Add new athlete…</option>');
  document.querySelectorAll('[data-logged-athlete-select]').forEach((select) => {
    const current = select.value;
    select.innerHTML = options.join('');
    if (current && names.includes(current)) {
      select.value = current;
    } else {
      select.value = '';
    }
    toggleLoggedAthleteInput(select);
  });
}

function renderTimeline(targetId = 'sessionTimeline', limit = 12) {
  const container = document.getElementById(targetId);
  if (!container) {
    return;
  }
  const sessions = state.sessions
    .filter((session) => {
      if (state.timelineFilter === 'all') {
        return true;
      }
      return (session.event || '').toLowerCase() === state.timelineFilter;
    })
    .slice(0, limit);
  if (sessions.length === 0) {
    container.innerHTML = '<p>No sessions logged yet.</p>';
    return;
  }
  container.innerHTML = sessions
    .map((session) => {
      const tags = (session.tags || []).map((tag) => `<span class="tag">${tag}</span>`).join('');
      const throws = session.throws?.length ? `${session.throws.length} throws` : '—';
      const action =
        targetId === 'sessionTimeline'
          ? `<button class="table-action ghost" data-delete-session="${session.id}">Delete</button>`
          : '';
      return `
        <div class="timeline-card">
          <div class="timeline-left">
            <p class="timeline-label">Best</p>
            <h3>${formatNumber(session.best)} m</h3>
            <p class="muted">${session.event || 'event'} · ${session.athlete || 'unknown'}</p>
            <small class="muted">${session.date}</small>
            <div class="tag-list">${tags}</div>
          </div>
          <div class="timeline-right">
            <div class="timeline-pill">RPE ${session.rpe ?? '—'}</div>
            <div class="timeline-pill">${throws}</div>
            <div class="timeline-pill">Load ${formatNumber(session.load, 1)} AU</div>
            ${action}
          </div>
        </div>
      `;
    })
    .join('');
}

function renderSessionsTable() {
  const table = document.getElementById('sessionsTable');
  if (!table) {
    return;
  }
  const tableElement = table.closest('table');
  const deletable = tableElement && tableElement.dataset.deletable === 'true';
  const query = state.searchQuery;
  const rows = state.sessions
    .filter((session) => {
      if (!query) {
        return true;
      }
      const haystack = (
        `${session.date} ${session.athlete} ${session.event} ${(session.tags || []).join(' ')}`
      ).toLowerCase();
      return haystack.includes(query);
    })
    .map((session) => {
      const actionCell = deletable
        ? `<td><button class="table-action" data-delete-session="${session.id}">Delete</button></td>`
        : '';
      return `
        <tr>
          <td>${session.date}</td>
          <td>${session.athlete || '—'}</td>
          <td>${session.event || '—'}</td>
          <td>${formatNumber(session.best)} m</td>
          <td>${session.throws?.length || 0}</td>
          <td>${formatNumber(session.load, 1)} AU</td>
          ${actionCell}
        </tr>
      `;
    });
  if (!rows.length) {
    const span = deletable ? 7 : 6;
    table.innerHTML = `<tr><td colspan="${span}">No sessions match your search.</td></tr>`;
  } else {
    table.innerHTML = rows.join('');
  }
}

function updateHeroHighlights() {
  const athletes = new Set(state.sessions.map((session) => (session.athlete || '').trim()).filter(Boolean));
  const heroAthletes = document.getElementById('heroAthletes');
  if (heroAthletes) {
    heroAthletes.textContent = athletes.size || '--';
  }
}

function renderTrainingIntel() {
  const intelList = document.getElementById('trainingIntelList');
  const readinessBoard = document.getElementById('readinessBoard');
  if (!state.summary || (!intelList && !readinessBoard)) {
    return;
  }
  if (intelList) {
    const pbGroups = state.summary.summary?.rows?.filter((row) => row.marker && row.marker.includes('PB')) || [];
    if (!pbGroups.length) {
      intelList.innerHTML = '<li>Log a few more sessions to surface PB contenders.</li>';
    } else {
      intelList.innerHTML = pbGroups
        .slice(0, 4)
        .map((row) => `<li><strong>${row.athlete}</strong> · ${row.event} — chasing PB in ${row.label}</li>`)
        .join('');
    }
  }
  renderRiskList('trainingRiskList');
  if (readinessBoard) {
    const totals = state.summary.summary?.totals;
    if (!totals || !totals.sessions) {
      readinessBoard.textContent = 'Need at least one session to compute readiness.';
    } else {
      const avgLoad = totals.load / totals.sessions;
      const avgRpe = totals.averageRpe != null ? formatNumber(totals.averageRpe, 1) : '—';
      const pb = state.summary.personalBest;
      readinessBoard.innerHTML = `
        <p><strong>${formatNumber(totals.load, 1)} AU</strong> total load this window</p>
        <p><strong>${formatNumber(avgLoad, 1)} AU</strong> average load / session · <strong>Avg RPE ${avgRpe}</strong></p>
        <p>${pb ? `Latest PB ${pb.date} at ${formatNumber(pb.best)} m.` : 'No PB logged in this window yet.'}</p>
      `;
    }
  }
}

function renderTrainingTimeline() {
  renderTimeline('trainingTimeline', 6);
}

function renderTagFocus() {
  const target = document.getElementById('tagFocus');
  if (!target) {
    return;
  }
  const counts = {};
  state.sessions.forEach((session) => {
    (session.tags || []).forEach((tag) => {
      const key = tag.toLowerCase();
      counts[key] = (counts[key] || 0) + 1;
    });
  });
  const entries = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  if (!entries.length) {
    target.innerHTML = '<li>No tags captured yet.</li>';
    return;
  }
  target.innerHTML = entries.map(([tag, value]) => `<li><strong>${tag}</strong> — ${value} mentions</li>`).join('');
}

async function refreshCliSummary() {
  const block = document.getElementById('cliSummaryText');
  if (!block) {
    return;
  }
  block.textContent = 'Loading CLI summary…';
  try {
    const response = await fetch(`/api/summary/table?group=${state.group}`);
    const data = await response.json();
    block.textContent = data.table || 'No sessions available to summarise.';
  } catch (error) {
    console.error(error);
    block.textContent = 'Unable to load CLI summary.';
  }
}

function showToast(message) {
  const toast = document.getElementById('appToast');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}

function setFormStatus(text) {
  const status = document.getElementById('formStatus');
  if (status) {
    status.textContent = text;
  }
}

function toggleChartPlaceholder(show) {
  const placeholder = document.getElementById('chartPlaceholder');
  if (!placeholder) {
    return;
  }
  placeholder.classList.toggle('visible', Boolean(show));
}

function formatNumber(value, digits = 2) {
  if (value == null || value === '') {
    return '—';
  }
  const number = Number(value);
  if (Number.isNaN(number)) {
    return '—';
  }
  return number.toFixed(digits);
}

function populateStrengthFilter() {
  const filter = document.getElementById('strengthFilter');
  if (!filter) {
    return;
  }
  const athletes = Array.from(new Set(state.strengthLogs.map((log) => log.athlete).filter(Boolean)));
  const options = ['all', ...athletes];
  filter.innerHTML = options
    .map((name) => `<option value="${name}">${name === 'all' ? 'All' : name}</option>`)
    .join('');
  if (!options.includes(state.strengthFilter)) {
    state.strengthFilter = 'all';
  }
  filter.value = state.strengthFilter;
}

function renderStrengthLogs() {
  const table = document.getElementById('strengthTable');
  if (!table) {
    return;
  }
  const filtered = state.strengthLogs.filter((log) => {
    if (state.strengthFilter === 'all') {
      return true;
    }
    return (log.athlete || '').toLowerCase() === state.strengthFilter.toLowerCase();
  });
  if (!filtered.length) {
    table.innerHTML = '<tr><td colspan="4">No workout sets recorded yet.</td></tr>';
    return;
  }

  const groups = {};
  filtered.forEach((log) => {
    const dateKey = (log.date?.split('T')[0] || log.date || '').split('T')[0];
    const athlete = (log.athlete || '—').toString().trim();
    const key = `${dateKey}::${athlete}`;
    if (!groups[key]) {
      groups[key] = { date: dateKey, athlete, sets: [] };
    }
    groups[key].sets.push(log);
  });

  const rows = Object.values(groups)
    .sort((a, b) => (a.date < b.date ? 1 : -1))
    .map((group, index) => {
      const setCount = group.sets.length;
      const summaryRow = `
        <tr class="strength-group-row">
          <td>${group.date}</td>
          <td>${group.athlete}</td>
          <td>${setCount}</td>
          <td><button type="button" class="table-action" data-toggle-strength="${index}">View sets</button></td>
        </tr>
      `;
      const detailRows = group.sets
        .map(
          (log) => `
            <tr>
              <td>${log.exercise || '—'}</td>
              <td>${formatNumber(log.load_kg, 1)} kg</td>
              <td>${log.reps ?? '—'}</td>
              <td>${log.notes || '—'}</td>
            </tr>
          `
        )
        .join('');
      const detailRow = `
        <tr class="strength-detail-row" data-strength-group="${index}">
          <td colspan="4">
            <table class="workout-table-inner">
              <thead>
                <tr>
                  <th>Exercise</th>
                  <th>Load</th>
                  <th>Reps</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                ${detailRows}
              </tbody>
            </table>
          </td>
        </tr>
      `;
      return summaryRow + detailRow;
    });

  table.innerHTML = rows.join('');
}

function renderReadinessSummary() {
  const block = document.getElementById('intelReadiness');
  if (!block) {
    return;
  }
  if (!state.summary) {
    block.textContent = 'Need more data to gauge readiness.';
    return;
  }
  const totals = state.summary.summary?.totals;
  if (!totals || !totals.sessions) {
    block.textContent = 'Log a few sessions to unlock readiness guidance.';
    return;
  }
  const avgLoad = totals.sessions ? totals.load / totals.sessions : 0;
  const avgRpe = totals.averageRpe != null ? formatNumber(totals.averageRpe, 1) : '—';
  const pb = state.summary.personalBest;
  block.innerHTML = `
    <p><strong>${formatNumber(avgLoad, 1)} AU</strong> average session load</p>
    <p><strong>${avgRpe}</strong> average RPE across the window</p>
    <p>${pb ? `Latest PB ${pb.date} at ${formatNumber(pb.best)} m.` : 'No PB logged in this window yet.'}</p>
  `;
}

async function deleteSession(sessionId) {
  if (!sessionId) {
    return;
  }
  if (!window.confirm('Delete this session? This cannot be undone.')) {
    return;
  }
  try {
    const response = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to delete session.');
    }
    showToast('Session deleted.');
    await Promise.all([loadSessions(), loadSummary()]);
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Unable to delete session.');
  }
}

function toggleLoggedAthleteInput(select) {
  const wrapper = select.closest('[data-athlete-select-wrapper]');
  if (!wrapper) {
    return;
  }
  const input = wrapper.querySelector('[data-logged-athlete-input]');
  if (select.value === '__custom') {
    wrapper.dataset.custom = 'true';
    if (input) {
      input.required = true;
      input.focus();
    }
  } else {
    wrapper.dataset.custom = 'false';
    if (input) {
      input.required = false;
    }
  }
}

function buildFormDataWithAthlete(form) {
  const formData = new FormData(form);
  const select = form.querySelector('[data-logged-athlete-select]');
  if (select) {
    let athleteValue = select.value;
    if (athleteValue === '__custom') {
      const customInput = form.querySelector('[data-logged-athlete-input]');
      athleteValue = customInput ? customInput.value.trim() : '';
    }
    formData.set('athlete', athleteValue);
  }
  return formData;
}

function toggleStrengthGroup(index, button) {
  const detailRow = document.querySelector(`.strength-detail-row[data-strength-group="${index}"]`);
  if (!detailRow) {
    return;
  }
  const isOpen = detailRow.classList.toggle('is-open');
  detailRow.style.display = isOpen ? 'table-row' : 'none';
  if (button) {
    button.textContent = isOpen ? 'Hide sets' : 'View sets';
  }
}

function logPlanRow(index) {
  const idx = Number(index);
  if (Number.isNaN(idx) || !state.plan[idx]) {
    return;
  }
  const athleteInput = document.getElementById('dailyPlanAthlete');
  const athlete = athleteInput ? athleteInput.value.trim() : '';
  if (!athlete) {
    showToast('Enter the athlete name for this plan first.');
    if (athleteInput) {
      athleteInput.focus();
    }
    return;
  }
  const row = document.querySelector(`.plan-log-row[data-plan-index="${idx}"]`);
  if (!row) {
    return;
  }
  const weightField = row.querySelector('input[name="weight_kg"]');
  const repsField = row.querySelector('input[name="reps"]');
  const weight = weightField ? weightField.value : '';
  const reps = repsField ? repsField.value : '';
  const item = state.plan[idx];
  const fd = new FormData();
  fd.set('athlete', athlete);
  fd.set('exercise', item.name || '');
  fd.set('weight_kg', weight || (item.target_weight_kg != null ? item.target_weight_kg : ''));
  fd.set('reps', reps || item.reps || item.target_reps || '');
  fd.set('auto_create', '1');
  submitLift(fd, { statusId: 'weightRoomLiftStatus' });
}

async function requestDailyPlan(formData) {
  const athlete = (formData.get('athlete') || '').trim();
  if (!athlete) {
    showToast('Select an athlete for the plan.');
    return;
  }
  const container = document.getElementById('dailyPlan');
  if (container) {
    container.textContent = 'Building plan...';
  }
  try {
    const response = await fetch('/api/training/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ athlete }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to build plan.');
    }
    state.plan = data.plan || [];
    renderDailyPlan();
  } catch (error) {
    console.error(error);
    if (container) {
      container.textContent = error.message || 'Unable to build plan.';
    }
  }
}

function renderDailyPlan() {
  const container = document.getElementById('dailyPlan');
  if (!container) {
    return;
  }
  if (!state.plan.length) {
    container.textContent = 'No plan available yet.';
    return;
  }
  const items = state.plan
    .map((item) => {
      const details = [];
      if (item.sets && item.reps) {
        details.push(`${item.sets} × ${item.reps}`);
      }
      if (item.target_weight_kg) {
        details.push(`${formatNumber(item.target_weight_kg, 1)} kg`);
      }
      if (item.target_distance_m) {
        details.push(`${formatNumber(item.target_distance_m)} m target`);
      }
      if (item.duration_minutes) {
        details.push(`${item.duration_minutes} min`);
      }
      if (item.duration_seconds) {
        details.push(`${item.duration_seconds}s`);
      }
      const meta = details.length ? `<span>${details.join(' • ')}</span>` : '';
      const notes = item.notes ? `<small>${item.notes}</small>` : '';
      return `<li><strong>${item.name}</strong>${meta}${notes}</li>`;
    })
    .join('');
  container.innerHTML = `<ul>${items}</ul>`;
  renderDailyPlanLogPanel();
}

async function submitLift(formData, options = {}) {
  const { statusId, form } = options;
  const status = statusId ? document.getElementById(statusId) : document.getElementById('liftStatus');
  if (status) {
    status.textContent = 'Logging set...';
  }
  const payload = Object.fromEntries(formData.entries());
  try {
    const response = await fetch('/api/training/log-lift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to log set.');
    }
    if (status) {
      status.textContent = data.message || 'Set logged.';
    }
    if (form) {
      form.reset();
    }
    showToast(data.message || 'Set logged.');
    await loadStrengthLogs();
    await ensureAthletes(true);
  } catch (error) {
    console.error(error);
    if (status) {
      status.textContent = error.message || 'Unable to log set.';
    }
  }
}

function renderDailyPlanLogPanel() {
  const panel = document.getElementById('dailyPlanLogPanel');
  if (!panel) {
    return;
  }
  if (!state.plan.length) {
    panel.textContent = 'Generate a plan to unlock quick logging.';
    return;
  }
  const lifts = state.plan.filter(
    (item) => item.is_strength || item.target_weight_kg != null || item.name
  );
  if (!lifts.length) {
    panel.textContent = 'No strength exercises found in today’s plan.';
    return;
  }
  const rows = lifts
    .map((item, index) => {
      const targetWeight = item.target_weight_kg != null ? formatNumber(item.target_weight_kg, 1) : '';
      const targetReps = item.reps || item.target_reps || '';
      const metaParts = [];
      if (item.sets && targetReps) {
        metaParts.push(`${item.sets} × ${targetReps}`);
      }
      if (targetWeight) {
        metaParts.push(`${targetWeight} kg`);
      }
      const meta = metaParts.length ? metaParts.join(' • ') : '';
      return `
        <li class="plan-log-row" data-plan-index="${index}">
          <div class="plan-log-main">
            <strong>${item.name}</strong>
            ${meta ? `<span>${meta}</span>` : ''}
          </div>
          <div class="plan-log-inputs">
            <label>
              <span>kg</span>
              <input type="number" step="0.5" name="weight_kg" value="${targetWeight || ''}" />
            </label>
            <label>
              <span>reps</span>
              <input type="number" name="reps" value="${targetReps || ''}" />
            </label>
            <button type="button" class="table-action" data-plan-log="${index}">Log</button>
          </div>
        </li>
      `;
    })
    .join('');
  panel.innerHTML = `<ul class="plan-log-list">${rows}</ul>`;
}

async function requestForecast(formData) {
  const output = document.getElementById('forecastOutput');
  if (output) {
    output.textContent = 'Running forecast...';
  }
  const athlete = (formData.get('athlete') || '').trim();
  const days = Number(formData.get('days') || 14);
  if (!athlete) {
    if (output) {
      output.textContent = 'Select an athlete first.';
    }
    return;
  }
  const metrics = ['throw_distance', 'session_load', 'bench_1rm', 'squat_1rm'];
  const results = [];
  let profileInfo = null;
  try {
    for (const metric of metrics) {
      const response = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ athlete, metric, days }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Unable to build forecast for ${metric}.`);
      }
      if (!profileInfo && data.profile) {
        profileInfo = data.profile;
      }
      results.push({ metric, forecast: data.forecast });
    }
    renderForecast(results, profileInfo);
  } catch (error) {
    console.error(error);
    if (output) {
      output.textContent = error.message || 'Unable to build forecast.';
    }
  }
}

async function requestWeeklyReport(formData) {
  const status = document.getElementById('weeklyReportStatus');
  if (status) {
    status.textContent = 'Generating PDF...';
  }
  const payload = Object.fromEntries(formData.entries());
  // Split events on comma if provided
  if (payload.events && typeof payload.events === 'string') {
    payload.events = payload.events
      .split(',')
      .map((e) => e.trim())
      .filter(Boolean);
  }
  try {
    const response = await fetch('/api/reports/weekly', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to generate report.');
    }
    const files = data.files || [];
    const message = files.length ? `Saved ${files.length} PDF(s): ${files.join(', ')}` : data.message;
    if (status) {
      status.textContent = message || 'Report generated.';
    }
    showToast(message || 'Report generated.');
  } catch (error) {
    console.error(error);
    if (status) {
      status.textContent = error.message || 'Unable to generate report.';
    }
  }
}

function renderForecast(resultSet, profile) {
  const output = document.getElementById('forecastOutput');
  if (!output) {
    return;
  }
  if (!resultSet || !resultSet.length) {
    output.textContent = 'No forecast available yet.';
    return;
  }
  const profileBlock = profile
    ? `
    <div class="forecast-profile">
      <h4>${profile.name}</h4>
      <p>Height: ${profile.height_cm ?? '—'} cm · Weight: ${profile.weight_kg ?? '—'} kg · BMI: ${
        profile.bmi != null ? formatNumber(profile.bmi, 1) : '—'
      }</p>
      <p>Bench 1RM: ${profile.bench_1rm_kg != null ? formatNumber(profile.bench_1rm_kg, 1) + ' kg' : '—'} ·
         Squat 1RM: ${profile.squat_1rm_kg != null ? formatNumber(profile.squat_1rm_kg, 1) + ' kg' : '—'}</p>
    </div>
  `
    : '';
  const sections = resultSet
    .map(({ metric, forecast }) => {
      const { model, trend, confidence, forecasts } = forecast || {};
      const header = `<p><strong>${metric.replace('_', ' ').toUpperCase()}</strong> — Model: ${
        model || 'n/a'
      } • Trend: ${trend || 'n/a'} • RMSE: ${confidence ?? 'n/a'}</p>`;
      const rows =
        (forecasts || [])
          .map(([date, value]) => `<tr><td>${date}</td><td>${formatNumber(value)}</td></tr>`)
          .join('') || '<tr><td colspan="2">Insufficient data.</td></tr>';
      return `${header}<table><thead><tr><th>Day</th><th>Forecast</th></tr></thead><tbody>${rows}</tbody></table>`;
    })
    .join('<hr/>');
  output.innerHTML = profileBlock + sections;
}
  const selectField = document.getElementById('athleteSelect');
  if (selectField) {
    selectField.innerHTML = state.athletes
      .map((athlete) => `<option value="${athlete.id}">${athlete.name}</option>`)
      .join('');
  }
