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
  lastReport: null,
  teams: [],
  teamFilter: 'all',
};

const page = document.body.dataset.page || null;
const controllers = {
  dashboard: initDashboard,
  sessions: initSessions,
  logs: initLogs,
  biomechanics: initBiomechanicsLab,
  analytics: initAnalytics,
  athletes: initAthletes,
  throwai: initTraining,
  weightroom: initWeightRoom,
  reports: initReports,
  quickstart: () => {},
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

async function ensureTeams(force = false) {
  if (!force && state.teams.length) {
    populateTeamFields();
    return state.teams;
  }
  try {
    const response = await fetch('/api/teams');
    const data = await response.json();
    state.teams = data.teams || [];
    populateTeamFields();
    return state.teams;
  } catch (error) {
    console.error(error);
    showToast('Unable to load teams.');
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
  const rows = state.athletes.filter((athlete) => {
    if (state.teamFilter && state.teamFilter !== 'all') {
      return (athlete.team || 'Unassigned').toLowerCase() === state.teamFilter.toLowerCase();
    }
    return true;
  });
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7">No athletes yet — add one to start logging sessions.</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (athlete) => `
        <tr data-athlete-id="${athlete.id}">
          <td>${athlete.name}</td>
          <td>${athlete.team || 'Unassigned'}</td>
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
  const teamField = document.getElementById('athleteTeam');
  if (!idField) {
    return;
  }
  idField.value = athlete.id;
  if (selectField) {
    selectField.value = String(athlete.id);
  }
  if (teamField) {
    const match = state.teams.find((team) => team.id === athlete.team_id) || null;
    const value = match ? match.id : athlete.team_id;
    teamField.value = value || '';
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
  clearFieldErrors(document.getElementById('athleteForm'));
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
    showToast('Athlete profile updated.', 'success');
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Unable to save profile.', 'error');
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

  document.querySelectorAll('[data-team-filter]').forEach((filter) => {
    filter.addEventListener('change', () => {
      state.teamFilter = filter.value || 'all';
      renderSessionsTable();
      renderTimeline();
    });
  });

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
        showToast('Event is required to log a session.', 'error');
        return;
      }
      setFormStatus('Saving...');
      const mode = event.submitter?.dataset?.mode || 'stay';
      const submitButton = event.submitter;
      if (submitButton) {
        submitButton.disabled = true;
        const originalText = submitButton.textContent;
        submitButton.dataset.originalText = originalText;
        submitButton.textContent = 'Saving...';
      }
      if (!validateSessionForm(form, payload)) {
        setFormStatus('Fix the highlighted fields');
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.textContent = submitButton.dataset.originalText || 'Save';
        }
        return;
      }
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
        const teamField = form.querySelector('select[name="team"]');
        const eventField = form.querySelector('input[name="event"]');
        if (teamField) {
          localStorage.setItem('lastTeam', teamField.value || '');
        }
        if (eventField) {
          localStorage.setItem('lastEvent', eventField.value || '');
        }
        if (mode === 'new') {
          const keep = {
            date: form.querySelector('input[name="date"]')?.value || '',
            team: teamField?.value || '',
            event: eventField?.value || '',
          };
          form.reset();
          if (keep.date) form.querySelector('input[name="date"]')?.setAttribute('value', keep.date);
          if (teamField && keep.team) teamField.value = keep.team;
          if (eventField && keep.event) eventField.value = keep.event;
          prefillWeekday(new Date().getDay());
          focusFirstField(form);
        } else {
          showToast('Session saved and analytics refreshed.', 'success');
        }
        await Promise.all([loadSummary(), loadSessions()]);
      } catch (error) {
        console.error(error);
        showToast(error.message || 'An unexpected error occurred.', 'error');
      } finally {
        setFormStatus('Ready');
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.textContent = submitButton.dataset.originalText || 'Save';
        }
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

  // Lightweight tooltip toggles for info-tip elements
  const tipTargets = document.querySelectorAll('.info-tip');
  tipTargets.forEach((tip) => {
    tip.setAttribute('tabindex', '0');
    tip.addEventListener('click', () => {
      const isOpen = tip.dataset.open === 'true';
      document.querySelectorAll('.info-tip[data-open="true"]').forEach((el) => (el.dataset.open = 'false'));
      tip.dataset.open = isOpen ? 'false' : 'true';
    });
    tip.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        tip.click();
      }
    });
  });
  document.addEventListener('click', (event) => {
    if (!event.target.closest('.info-tip')) {
      document.querySelectorAll('.info-tip[data-open="true"]').forEach((el) => (el.dataset.open = 'false'));
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

  const filterToggle = document.getElementById('filterToggle');
  if (filterToggle) {
    const bar = document.querySelector('.filter-bar.collapsible');
    filterToggle.addEventListener('click', () => {
      if (bar) {
        bar.classList.toggle('is-open');
      }
    });
  }

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

  const templateSelect = document.getElementById('sessionTemplate');
  if (templateSelect) {
    templateSelect.addEventListener('change', () => applySessionTemplate(templateSelect.value));
  }
  document.querySelectorAll('[data-template]').forEach((button) => {
    button.addEventListener('click', () => applySessionTemplate(button.dataset.template));
  });
  const openTemplate = document.getElementById('openTemplateSelect');
  if (openTemplate && templateSelect) {
    openTemplate.addEventListener('click', () => {
      templateSelect.scrollIntoView({ behavior: 'smooth', block: 'center' });
      templateSelect.focus();
    });
  }
  const duplicateBtn = document.getElementById('duplicateSession');
  if (duplicateBtn) {
    duplicateBtn.addEventListener('click', () => duplicateLastSession());
  }
  const quickAddBtn = document.getElementById('quickAddSession');
  if (quickAddBtn) {
    quickAddBtn.addEventListener('click', () => prefillWeekday(new Date().getDay()));
  }
  const prefillMonday = document.getElementById('prefillMonday');
  if (prefillMonday) {
    prefillMonday.addEventListener('click', () => prefillWeekday(1));
  }
  const reportDownload = document.getElementById('reportDownload');
  if (reportDownload) {
    reportDownload.addEventListener('click', () => handleReportAction('download'));
  }
  const reportEmail = document.getElementById('reportEmail');
  if (reportEmail) {
    reportEmail.addEventListener('click', () => handleReportAction('email'));
  }

  const rosterImportForm = document.getElementById('rosterImportForm');
  if (rosterImportForm) {
    rosterImportForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      await importCsv(rosterImportForm, '/api/import/roster', 'rosterImportStatus');
    });
  }
  const sessionImportForm = document.getElementById('sessionImportForm');
  if (sessionImportForm) {
    sessionImportForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      await importCsv(sessionImportForm, '/api/import/sessions', 'sessionImportStatus');
      await loadSessions();
    });
  }

  const teamForm = document.getElementById('teamForm');
  if (teamForm) {
    teamForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(teamForm);
      const name = formData.get('name');
      if (!name) return;
      try {
        const response = await fetch('/api/teams', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Unable to create team.');
        }
        showToast(`Created team ${name}.`);
        teamForm.reset();
        await ensureTeams(true);
      } catch (error) {
        showToast(error.message || 'Unable to create team.');
      }
    });
  }
}

async function initDashboard() {
  await Promise.all([loadSummary(), loadSessions(), ensureTeams()]);
  initOnboardingTour();
}

async function initSessions() {
  await Promise.all([loadSessions(), ensureAthletes(), ensureTeams()]);
  const dateInput = document.querySelector('#sessionForm input[name="date"]');
  if (dateInput && !dateInput.value) {
    const today = new Date();
    dateInput.value = formatInputDate(today);
  }
  applyLastSelections();
  focusFirstField(document.getElementById('sessionForm'));
}

async function initLogs() {
  await Promise.all([loadSummary(), loadSessions(), loadStrengthLogs(), ensureAthletes(), ensureTeams()]);
  renderStrengthLogs();
}

async function initAnalytics() {
  await Promise.all([loadSummary(), loadSessions(), ensureAthletes(), ensureTeams()]);
  await refreshCliSummary();
}

async function initTraining() {
  await Promise.all([loadSummary(), loadSessions(), ensureAthletes(), ensureTeams()]);
}

async function initReports() {
  await Promise.all([loadSummary(), refreshCliSummary(), ensureAthletes(), ensureTeams()]);
  prefillReportFormDefaults();
  updateLastReportCard();
  const form = document.getElementById('weeklyReportForm');
  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      await requestWeeklyReport(new FormData(form));
    });
  }
}

async function initBiomechanicsLab() {
  await Promise.all([ensureAthletes()]);

  const dropzone = document.getElementById('biomechDropzone');
  const browseButton = document.getElementById('biomechBrowseButton');
  const fileInput = document.getElementById('biomechFileInput');
  const analyzeButton = document.getElementById('biomechAnalyzeButton');
  const resetButton = document.getElementById('biomechResetButton');
  const athleteInput = document.getElementById('biomechLabAthlete');
  const dateInput = document.getElementById('biomechLabDate');
  const bestInput = document.getElementById('biomechLabBest');
  const selectedFileLabel = document.getElementById('biomechSelectedFile');
  const sessionLink = document.getElementById('biomechSessionLink');
  const resultsPanel = document.getElementById('biomechResultsPanel');
  const viewerRoot = document.getElementById('biomechanicsViewerRoot');
  const progressPanel = document.getElementById('biomechLabProgress');
  const progressFill = document.getElementById('biomechLabProgressFill');
  const progressStatus = document.getElementById('biomechLabStatus');

  if (!dropzone || !fileInput || !analyzeButton || !resetButton || !athleteInput || !dateInput || !selectedFileLabel) {
    return;
  }

  const defaultAnalyzeLabel = analyzeButton.textContent || 'Analyze';

  if (dateInput && !dateInput.value) {
    const today = new Date();
    dateInput.value = formatInputDate(today);
  }

  let selectedFile = null;
  let busy = false;
  let currentSessionId = null;
  let progressTimer = null;

  function clearProgressTimer() {
    if (progressTimer) {
      window.clearInterval(progressTimer);
      progressTimer = null;
    }
  }

  function setProgress(message, percent) {
    if (!progressPanel || !progressFill || !progressStatus) {
      return;
    }
    progressPanel.style.display = 'block';
    if (message) {
      progressStatus.textContent = message;
    }
    const pct = Math.max(0, Math.min(100, Number(percent) || 0));
    progressFill.style.width = `${pct}%`;
    const bar = progressPanel.querySelector('.progress-bar');
    if (bar) {
      bar.setAttribute('aria-valuenow', String(pct));
    }
  }

  function hideProgress() {
    if (!progressPanel || !progressFill || !progressStatus) {
      return;
    }
    progressPanel.style.display = 'none';
    progressFill.style.width = '0%';
    progressStatus.textContent = 'Ready.';
    const bar = progressPanel.querySelector('.progress-bar');
    if (bar) {
      bar.setAttribute('aria-valuenow', '0');
    }
  }

  function updateControls() {
    const fileOk = Boolean(selectedFile);
    analyzeButton.disabled = busy || !fileOk;
    resetButton.disabled = busy || (!selectedFile && !currentSessionId);
    analyzeButton.textContent = busy ? 'Analyzing…' : defaultAnalyzeLabel;
  }

  function setSelectedFile(file) {
    selectedFile = file || null;
    if (!selectedFile) {
      selectedFileLabel.textContent = 'No file selected.';
      return;
    }
    const sizeMb = selectedFile.size ? selectedFile.size / (1024 * 1024) : 0;
    selectedFileLabel.textContent = `${selectedFile.name} (${formatNumber(sizeMb, 1)} MB)`;
  }

  function resetUi() {
    clearProgressTimer();
    busy = false;
    selectedFile = null;
    currentSessionId = null;
    setSelectedFile(null);
    fileInput.value = '';
    hideProgress();
    if (sessionLink) sessionLink.innerHTML = '';
    if (resultsPanel) resultsPanel.style.display = 'none';
    if (viewerRoot) {
      viewerRoot.dataset.sessionId = '';
      viewerRoot.dataset.athleteName = '';
      if (window.BiomechanicsViewer && typeof window.BiomechanicsViewer.unmount === 'function') {
        window.BiomechanicsViewer.unmount(viewerRoot);
      } else {
        viewerRoot.innerHTML = '';
      }
    }
    updateControls();
  }

  function isSupportedVideo(file) {
    const name = String(file?.name || '').toLowerCase();
    return name.endsWith('.mp4') || name.endsWith('.mov') || name.endsWith('.avi') || name.endsWith('.mkv');
  }

  function uploadWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
      xhr.responseType = 'json';

      xhr.upload.addEventListener('progress', (event) => {
        if (!event.lengthComputable) return;
        const ratio = event.total > 0 ? event.loaded / event.total : 0;
        if (typeof onProgress === 'function') {
          onProgress(Math.max(0, Math.min(1, ratio)));
        }
      });

      xhr.addEventListener('load', () => {
        const status = xhr.status;
        const payload = xhr.response || null;
        if (status >= 200 && status < 300) {
          resolve(payload);
          return;
        }
        reject(new Error(payload?.error || 'Unable to upload video.'));
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error while uploading video.'));
      });

      xhr.send(formData);
    });
  }

  function startProgressPolling(sessionId) {
    clearProgressTimer();

    const tick = async () => {
      try {
        const response = await fetch(`/api/sessions/${sessionId}/biomechanics/progress`);
        const data = await response.json();
        if (!response.ok) {
          return;
        }

        const pctRaw = Number(data?.percent_complete);
        const pct = Number.isFinite(pctRaw) ? Math.max(0, Math.min(100, pctRaw)) : 0;
        const status = String(data?.status || '').toLowerCase();

        const mapped = 35 + (pct / 100) * 65;
        const label =
          status === 'complete'
            ? 'Complete.'
            : status === 'error'
              ? 'Failed.'
              : `Processing… ${pct.toFixed(0)}%`;
        setProgress(label, status === 'complete' ? 100 : mapped);

        if (status === 'complete') {
          clearProgressTimer();
          busy = false;
          updateControls();
          showToast('Biomechanics analysis complete.', 'success');
        }

        if (status === 'error') {
          clearProgressTimer();
          busy = false;
          updateControls();
          const msg = data?.error_message || 'Biomechanics processing failed.';
          showToast(msg, 'error');
          setProgress(`Failed: ${msg}`, 100);
        }
      } catch (err) {
        // Ignore transient poll errors.
      }
    };

    // Kick once immediately, then poll.
    tick();
    progressTimer = window.setInterval(tick, 1500);
  }

  async function startAnalysis() {
    if (busy) return;

    if (!selectedFile) {
      showToast('Drop a video file first.', 'error');
      return;
    }
    if (!isSupportedVideo(selectedFile)) {
      showToast('Unsupported file type. Use mp4/mov/avi/mkv.', 'error');
      return;
    }
    if (selectedFile.size > 100 * 1024 * 1024) {
      showToast('File too large (max 100MB).', 'error');
      return;
    }

    const athleteName = (athleteInput.value || '').trim() || 'unassigned';

    busy = true;
    updateControls();
    clearProgressTimer();
    setProgress('Creating session…', 5);

    try {
      sessionLink && (sessionLink.textContent = 'Creating session…');
      const dateValue = (dateInput.value || '').trim();
      const bestRaw = bestInput ? (bestInput.value || '').trim() : '';
      const best = bestRaw ? Number(bestRaw) : 0;
      const payload = {
        athlete: athleteName,
        date: dateValue || undefined,
        event: 'javelin',
        best: Number.isFinite(best) ? best : 0,
        throws: null,
        rpe: null,
        duration_minutes: 0,
        tags: 'biomechanics',
        notes: 'Biomechanics video analysis session.',
      };

      const sessionResp = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const sessionData = await sessionResp.json();
      if (!sessionResp.ok) {
        throw new Error(sessionData.error || 'Unable to create session.');
      }
      const session = sessionData.session || {};
      const sessionId = String(session.id || '');
      if (!sessionId) {
        throw new Error('Session id missing from server response.');
      }
      currentSessionId = sessionId;

      sessionLink &&
        (sessionLink.innerHTML = `Session <a href="/sessions/${sessionId}">${sessionId}</a> · Uploading video…`);
      setProgress('Uploading video…', 15);

      const formData = new FormData();
      formData.append('file', selectedFile);

      await uploadWithProgress(`/api/sessions/${sessionId}/upload-video`, formData, (ratio) => {
        const pct = 15 + ratio * 20;
        setProgress('Uploading video…', pct);
      });

      showToast('Video uploaded. Biomechanics processing started.', 'success');
      sessionLink &&
        (sessionLink.innerHTML = `Processing… <a href="/sessions/${sessionId}">Open session details</a>`);
      setProgress('Processing video…', 35);

      if (resultsPanel) resultsPanel.style.display = 'block';
      if (viewerRoot) {
        viewerRoot.dataset.sessionId = sessionId;
        viewerRoot.dataset.athleteName = athleteName;
        if (window.BiomechanicsViewer && typeof window.BiomechanicsViewer.mount === 'function') {
          window.BiomechanicsViewer.mount(viewerRoot, { sessionId, athleteName });
        }
      }

      resultsPanel?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      startProgressPolling(sessionId);
    } catch (error) {
      console.error(error);
      showToast(error.message || 'Unable to start biomechanics analysis.', 'error');
      sessionLink && (sessionLink.textContent = '');
      setProgress(`Error: ${error.message || 'Unable to start.'}`, 100);
      busy = false;
      updateControls();
    }
  }

  function pickFile() {
    fileInput.click();
  }

  browseButton?.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    pickFile();
  });
  dropzone.addEventListener('click', pickFile);
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickFile();
    }
  });

  athleteInput.addEventListener('input', updateControls);
  dateInput.addEventListener('change', updateControls);
  bestInput?.addEventListener('input', updateControls);

  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
    setSelectedFile(file);
    updateControls();
  });

  analyzeButton.addEventListener('click', startAnalysis);
  resetButton.addEventListener('click', resetUi);

  const prevent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  dropzone.addEventListener('dragenter', (event) => {
    prevent(event);
    dropzone.classList.add('is-dragover');
  });
  dropzone.addEventListener('dragover', (event) => {
    prevent(event);
    dropzone.classList.add('is-dragover');
  });
  dropzone.addEventListener('dragleave', (event) => {
    prevent(event);
    dropzone.classList.remove('is-dragover');
  });
  dropzone.addEventListener('drop', (event) => {
    prevent(event);
    dropzone.classList.remove('is-dragover');
    const file =
      event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length
        ? event.dataTransfer.files[0]
        : null;
    if (!file) return;
    setSelectedFile(file);
    updateControls();
  });

  hideProgress();
  updateControls();
}

async function initWeightRoom() {
  await Promise.all([ensureAthletes(), loadSessions(), loadStrengthLogs(), ensureTeams()]);
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
      clearFieldErrors(form);
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
    renderWeekPlanner();
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
  updateLoadRiskBanner();
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
          borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.15)',
      tension: 0.4,
      borderWidth: 2,
      yAxisID: 'distance',
    },
    {
      label: 'Median best',
          data: median,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.18)',
          tension: 0.4,
      borderDash: [6, 6],
      borderWidth: 2,
      yAxisID: 'distance',
    },
    {
      label: 'Throw volume (count)',
      data: volume,
      type: 'bar',
      backgroundColor: 'rgba(249, 115, 22, 0.2)',
      borderColor: '#f97316',
      yAxisID: 'volume',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        distance: {
          beginAtZero: true,
          grid: { color: 'rgba(15,23,42,0.08)' },
          ticks: { color: '#475569' },
          title: { display: true, text: 'Best distance (m)', color: '#475569' },
        },
        volume: {
          beginAtZero: true,
          position: 'right',
          grid: { display: false },
          ticks: { color: '#475569' },
          title: { display: true, text: 'Throw volume (count)', color: '#475569' },
        },
        x: { grid: { display: false }, ticks: { color: '#475569' } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.92)',
          borderColor: 'rgba(0,0,0,0.08)',
          borderWidth: 1,
        },
      },
    },
  });
}

function renderSummaryTable() {
  if (!state.summary) {
    return;
  }
  const table = document.getElementById('summaryTable');
  const rows =
    state.summaryTab === 'all'
      ? state.summary.summary?.rollupRows
      : state.summary.summary?.rows;
  const heroAlerts = document.getElementById('heroAlerts');
  const riskTile = document.getElementById('riskAlertCount');
  const riskCount = rows && rows.length ? rows.filter((row) => row.risk).length : 0;
  if (heroAlerts) {
    heroAlerts.textContent = riskCount;
  }
  if (riskTile) {
    riskTile.textContent = riskCount;
  }
  if (!table) {
    updateLoadRiskBanner();
    return;
  }
  if (!rows || rows.length === 0) {
    table.innerHTML = `<tr><td colspan="12">No sessions logged yet.</td></tr>`;
    updateLoadRiskBanner();
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
          <td>${formatNumber(row.monotony, 2)}</td>
          <td>${formatNumber(row.strain, 0)}</td>
          <td>${riskBadge}</td>
        </tr>
      `;
    })
    .join('');
  updateLoadRiskBanner();
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

function populateTeamFields() {
  const optionsByName = {
    default: ['<option value="all">All teams</option>', '<option value="">Unassigned</option>'],
    session: ['<option value="Unassigned">Unassigned</option>'],
    teamId: ['<option value="">Unassigned</option>'],
  };
  state.teams.forEach((team) => {
    optionsByName.default.push(`<option value="${team.name}">${team.name}</option>`);
    optionsByName.session.push(`<option value="${team.name}">${team.name}</option>`);
    optionsByName.teamId.push(`<option value="${team.id}">${team.name}</option>`);
  });

  document.querySelectorAll('[data-team-select]').forEach((select) => {
    const current = select.value;
    let opts = optionsByName.default;
    if (select.name === 'team') {
      opts = optionsByName.session;
    } else if (select.name === 'team_id') {
      opts = optionsByName.teamId;
    }
    select.innerHTML = opts.join('');
    if (current) {
      select.value = current;
    }
  });
  document.querySelectorAll('[data-team-filter]').forEach((filter) => {
    filter.innerHTML = optionsByName.default.join('');
    filter.value = state.teamFilter || 'all';
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
    .filter((session) => {
      if (state.teamFilter && state.teamFilter !== 'all') {
        return (session.team || 'Unassigned').toLowerCase() === state.teamFilter.toLowerCase();
      }
      return true;
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
      if (state.timelineFilter !== 'all') {
        return (session.event || '').toLowerCase() === state.timelineFilter;
      }
      return true;
    })
    .filter((session) => {
      if (state.teamFilter && state.teamFilter !== 'all') {
        return (session.team || 'Unassigned').toLowerCase() === state.teamFilter.toLowerCase();
      }
      return true;
    })
    .filter((session) => {
      if (!query) {
        return true;
      }
      const haystack = (
        `${session.date} ${session.athlete} ${session.event} ${(session.tags || []).join(' ')} ${session.team || ''}`
      ).toLowerCase();
      return haystack.includes(query);
    })
    .map((session) => {
      const actionCell = deletable
        ? `<td><button class="table-action" data-delete-session="${session.id}">Delete</button></td>`
        : '';
      const tags = (session.tags || []).join(', ');
      const volume = session.throws?.length || 0;
      const rpe = session.rpe ?? '—';
      const biomechRaw = (session.biomechanics_status || '').toString().trim().toLowerCase();
      const biomechLabel = biomechRaw ? biomechRaw[0].toUpperCase() + biomechRaw.slice(1) : '—';
      const biomechCell = session.id
        ? `<a class="table-action ghost" href="/sessions/${session.id}">${biomechLabel}</a>`
        : biomechLabel;
      return `
        <tr>
          <td data-label="Date">${session.date}</td>
          <td data-label="Athlete">${session.athlete || '—'}</td>
          <td data-label="Team">${session.team || 'Unassigned'}</td>
          <td data-label="Event">${session.event || '—'}</td>
          <td data-label="Best">${formatNumber(session.best)} m</td>
          <td data-label="Volume">${volume}</td>
          <td data-label="RPE">${rpe}</td>
          <td data-label="Tags">${tags || '—'}</td>
          <td data-label="Load">${formatNumber(session.load, 1)} AU</td>
          <td data-label="Biomechanics">${biomechCell}</td>
          ${actionCell}
        </tr>
      `;
    });
  if (!rows.length) {
    const span = deletable ? 11 : 10;
    table.innerHTML = `<tr><td colspan="${span}">No sessions match your search.</td></tr>`;
  } else {
    table.innerHTML = rows.join('');
  }
}

function getWeekDates(date = new Date()) {
  const today = new Date(date);
  const day = today.getDay();
  const diff = day === 0 ? -6 : 1 - day; // Monday start
  const monday = new Date(today);
  monday.setHours(0, 0, 0, 0);
  monday.setDate(today.getDate() + diff);
  return Array.from({ length: 7 }, (_, index) => {
    const d = new Date(monday);
    d.setDate(monday.getDate() + index);
    return d;
  });
}

function formatInputDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function renderWeekPlanner() {
  const strip = document.getElementById('weekPlanner');
  if (!strip) return;
  const dates = getWeekDates();
  const weekKeys = new Set(dates.map((d) => formatInputDate(d)));
  const sessionsByDay = new Set();
  state.sessions.forEach((session) => {
    if (!session.date) return;
    const date = new Date(session.date);
    if (Number.isNaN(date.getTime())) return;
    const key = formatInputDate(date);
    if (weekKeys.has(key)) {
      sessionsByDay.add(key);
    }
  });
  strip.querySelectorAll('.week-day').forEach((button) => {
    const weekday = Number(button.dataset.weekday);
    const dateForDay = dates.find((d) => d.getDay() === weekday);
    const dateLabel = button.querySelector('.week-date');
    if (dateForDay && dateLabel) {
      dateLabel.textContent = dateForDay.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      button.dataset.dateValue = formatInputDate(dateForDay);
      const hasSession = sessionsByDay.has(button.dataset.dateValue);
      button.classList.toggle('has-session', hasSession);
    }
    button.onclick = () => prefillWeekday(weekday);
  });
  updateSessionsEmptyState(sessionsByDay.size > 0);
}

function prefillWeekday(weekday) {
  const dates = getWeekDates();
  const target = dates.find((d) => d.getDay() === weekday);
  const dateInput = document.querySelector('#sessionForm input[name="date"]');
  if (target && dateInput) {
    dateInput.value = formatInputDate(target);
    dateInput.focus();
  }
}

function applyLastSelections() {
  const teamField = document.querySelector('#sessionForm select[name="team"]');
  const eventField = document.querySelector('#sessionForm input[name="event"]');
  if (teamField) {
    const lastTeam = localStorage.getItem('lastTeam');
    if (lastTeam) {
      teamField.value = lastTeam;
    }
  }
  if (eventField) {
    const lastEvent = localStorage.getItem('lastEvent');
    if (lastEvent) {
      eventField.value = lastEvent;
    }
  }
  const rpeField = document.querySelector('#sessionForm input[name="rpe"]');
  if (rpeField && !rpeField.value) {
    rpeField.value = '7';
  }
}

function updateSessionsEmptyState(hasSessionsThisWeek) {
  const empty = document.getElementById('sessionsEmptyState');
  if (!empty) return;
  empty.style.display = hasSessionsThisWeek ? 'none' : 'flex';
}

function applySessionTemplate(template) {
  if (!template) return;
  const best = document.querySelector('input[name="best"]');
  const throwsField = document.querySelector('input[name="throws"]');
  const rpe = document.querySelector('input[name="rpe"]');
  const duration = document.querySelector('input[name="duration_minutes"]');
  const tags = document.querySelector('input[name="tags"]');
  const notes = document.querySelector('textarea[name="notes"]');
  const technique = document.querySelector('input[name="technique"]');
  const presets = {
    tech: {
      rpe: '5',
      duration: '45',
      tags: 'technique, drills',
      notes: 'Technical focus: positions, rhythm, light intensity.',
      technique: 'Technical session',
    },
    heavy: {
      rpe: '8',
      duration: '75',
      tags: 'strength, heavy throws',
      notes: 'Heavy implements, lower volume. Track readiness.',
      technique: 'Heavy throws',
    },
    comp: {
      rpe: '9',
      duration: '90',
      tags: 'competition, meet',
      notes: 'Meet day. Capture conditions and best marks.',
      technique: 'Competition',
    },
  };
  if (template === 'last') {
    duplicateLastSession();
    return;
  }
  if (template === 'clear') {
    [best, throwsField, rpe, duration, tags, notes, technique].forEach((field) => field && (field.value = ''));
    return;
  }
  const preset = presets[template];
  if (!preset) return;
  rpe && (rpe.value = preset.rpe);
  duration && (duration.value = preset.duration);
  tags && (tags.value = preset.tags);
  notes && (notes.value = preset.notes);
  technique && (technique.value = preset.technique);
  if (best && !best.value) {
    best.focus();
  }
}

function duplicateLastSession() {
  if (!state.sessions.length) {
    showToast('No previous sessions to duplicate.');
    return;
  }
  const sorted = [...state.sessions].sort((a, b) => {
    const aDate = new Date(a.date || 0).getTime();
    const bDate = new Date(b.date || 0).getTime();
    return bDate - aDate;
  });
  const session = sorted[0];
  const form = document.getElementById('sessionForm');
  if (!form) return;
  const map = {
    athlete: session.athlete || '',
    event: session.event || '',
    date: session.date ? session.date.split('T')[0] : '',
    best: session.best ?? '',
    throws: Array.isArray(session.throws) ? session.throws.join(', ') : session.throws ?? '',
    rpe: session.rpe ?? '',
    duration_minutes: session.duration_minutes ?? '',
    team: session.team ?? '',
    implement_weight_kg: session.implement_weight_kg ?? '',
    technique: session.technique ?? '',
    fouls: session.fouls ?? '',
    tags: session.tags ? session.tags.join(', ') : '',
    notes: session.notes ?? '',
  };
  Object.entries(map).forEach(([name, value]) => {
    const field = form.querySelector(`[name="${name}"]`);
    if (field) {
      field.value = value;
    }
  });
  const select = form.querySelector('[data-logged-athlete-select]');
  if (select) {
    select.value = map.athlete;
    toggleLoggedAthleteInput(select);
  }
  showToast('Duplicated last session—review and save.');
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
      const latestRow =
        state.summary.summary?.rows && state.summary.summary.rows.length
          ? state.summary.summary.rows[state.summary.summary.rows.length - 1]
          : null;
      const acwr = latestRow ? formatNumber(latestRow.acwrRolling) : '—';
      const monotony = latestRow ? formatNumber(latestRow.monotony, 2) : '—';
      const strain = latestRow ? formatNumber(latestRow.strain, 0) : '—';
      readinessBoard.innerHTML = `
        <p><strong>${formatNumber(totals.load, 1)} AU</strong> total load this window</p>
        <p><strong>${formatNumber(avgLoad, 1)} AU</strong> average load / session · <strong>Avg RPE ${avgRpe}</strong></p>
        <p>ACWR ${acwr} · Monotony ${monotony} · Strain ${strain}</p>
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

function showToast(message, type = 'info') {
  const toast = document.getElementById('appToast');
  if (!toast) return;
  toast.classList.remove('success', 'error');
  if (type === 'success') toast.classList.add('success');
  if (type === 'error') toast.classList.add('error');
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

function validateSessionForm(form, payload) {
  clearFieldErrors(form);
  const errors = [];
  if (!payload.athlete) {
    errors.push({ field: 'athlete', message: 'Athlete is required.' });
  }
  if (!payload.date) {
    errors.push({ field: 'date', message: 'Date is required.' });
  }
  if (!payload.event) {
    errors.push({ field: 'event', message: 'Event is required.' });
  }
  if (payload.rpe) {
    const rpeValue = Number(payload.rpe);
    if (Number.isNaN(rpeValue) || rpeValue < 1 || rpeValue > 10) {
      errors.push({ field: 'rpe', message: 'Enter an RPE between 1 and 10.' });
    }
  }
  if (!payload.rpe) {
    errors.push({ field: 'rpe', message: 'RPE helps calculate load.' });
  }
  errors.forEach((error) => showFieldError(form, error.field, error.message));
  if (errors.length) {
    const first = form.querySelector(`[name="${errors[0].field}"]`);
    first?.focus();
    return false;
  }
  return true;
}

function showFieldError(form, fieldName, message) {
  const target = form.querySelector(`[data-error-for="${fieldName}"]`);
  const input = form.querySelector(`[name="${fieldName}"]`);
  if (target) {
    target.textContent = message;
  }
  if (input) {
    input.classList.add('input-error');
  }
}

function clearFieldErrors(form) {
  form.querySelectorAll('.field-error').forEach((el) => (el.textContent = ''));
  form.querySelectorAll('.input-error').forEach((el) => el.classList.remove('input-error'));
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
    table.innerHTML = '<tr><td colspan="6">No workout sets recorded yet.</td></tr>';
    return;
  }

  const rows = filtered
    .sort((a, b) => ((a.date || '') < (b.date || '') ? 1 : -1))
    .map((log) => {
      const date = (log.date?.split('T')[0] || log.date || '').split('T')[0] || '—';
      const sets = log.sets ?? 1;
      const reps = log.reps ?? '—';
      const volume = `${sets} × ${reps}`;
      return `
        <tr>
          <td data-label="Date">${date}</td>
          <td data-label="Athlete">${log.athlete || '—'}</td>
          <td data-label="Lift">${log.exercise || '—'}</td>
          <td data-label="Sets × reps">${volume}</td>
          <td data-label="Load">${formatNumber(log.load_kg, 1)} kg</td>
          <td data-label="Notes">${log.notes || '—'}</td>
        </tr>
      `;
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
    block.textContent = 'Not enough data to estimate readiness. Log 3 sessions this week.';
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

function updateLoadRiskBanner() {
  const loadEl = document.getElementById('bannerLoadValue');
  const riskEl = document.getElementById('bannerRiskLevel');
  if (!loadEl && !riskEl) {
    return;
  }
  const heroLoad = document.getElementById('heroLoad') || document.getElementById('statLoad');
  const heroAlerts = document.getElementById('heroAlerts') || document.getElementById('riskAlertCount');
  const loadText = heroLoad ? heroLoad.textContent || '--' : '--';
  const riskCount = heroAlerts ? Number(heroAlerts.textContent || 0) : 0;
  let riskLevel = 'Low';
  if (riskCount > 3) riskLevel = 'High';
  else if (riskCount > 0) riskLevel = 'Medium';
  if (loadEl) {
    loadEl.textContent = loadText;
  }
  if (riskEl) {
    riskEl.textContent = Number.isNaN(riskCount) ? riskLevel : `${riskLevel} (${riskCount})`;
  }
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
    state.lastReport = {
      message: message || 'Report generated.',
      athlete: payload.athlete || 'All athletes',
      weekEnding: payload.week_ending || '',
      timestamp: new Date().toISOString(),
    };
    updateLastReportCard();
    showToast(message || 'Report generated.');
  } catch (error) {
    console.error(error);
    if (status) {
      status.textContent = error.message || 'Unable to generate report.';
    }
  }
}

function prefillReportFormDefaults() {
  const form = document.getElementById('weeklyReportForm');
  if (!form) return;
  const weekEnding = form.querySelector('input[name="week_ending"]');
  if (weekEnding && !weekEnding.value) {
    weekEnding.value = getUpcomingSunday();
  }
  const athleteSelect = form.querySelector('select[name="athlete"]');
  if (athleteSelect && !athleteSelect.value) {
    const first = Array.from(athleteSelect.options).find((opt) => opt.value);
    if (first) {
      athleteSelect.value = first.value;
    }
  }
  const teamSelect = form.querySelector('select[name="team"]');
  if (teamSelect && !teamSelect.value && state.teams.length === 1) {
    teamSelect.value = state.teams[0].name;
  }
}

function getUpcomingSunday(date = new Date()) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = (7 - day) % 7;
  d.setDate(d.getDate() + diff);
  return formatInputDate(d);
}

function updateLastReportCard() {
  const cardRecipient = document.getElementById('reportRecipient');
  const cardTime = document.getElementById('reportTimestamp');
  if (!cardRecipient && !cardTime) return;
  if (!state.lastReport) {
    if (cardRecipient) {
      cardRecipient.textContent = 'No reports yet.';
    }
    if (cardTime) {
      cardTime.textContent = '';
    }
    return;
  }
  if (cardRecipient) {
    cardRecipient.textContent = `${state.lastReport.athlete} · Week ending ${state.lastReport.weekEnding || '—'}`;
  }
  if (cardTime) {
    const time = new Date(state.lastReport.timestamp || new Date());
    cardTime.textContent = `Generated ${time.toLocaleString()}`;
  }
}

function handleReportAction(type) {
  if (!state.lastReport) {
    showToast('Generate a report first.');
    return;
  }
  const action = type === 'email' ? 'Emailing' : 'Downloading';
  showToast(`${action} latest report: ${state.lastReport.message || ''}`);
}

function initOnboardingTour() {
  const startPanel = document.getElementById('startHerePanel');
  if (!startPanel || localStorage.getItem('coachTourDone') === '1') {
    return;
  }
  const steps = [
    {
      title: 'Navigate your control center',
      body: 'Use the sidebar to jump between Dashboard, Sessions, Analytics, and Reports.',
      selector: '.sidebar__nav',
    },
    {
      title: 'Log a session',
      body: 'Head to Sessions to capture best throws, RPE, tags, and notes each day.',
      selector: '#sessionForm',
    },
    {
      title: 'See load & export reports',
      body: 'Check Analytics for load/readiness and generate weekly PDFs from Reports.',
      selector: '#load-readiness',
    },
  ].filter(Boolean);
  if (!steps.length) return;

  const overlay = document.createElement('div');
  overlay.className = 'onboarding-overlay';
  overlay.innerHTML = `
    <div class="onboarding-card">
      <h4></h4>
      <p class="muted"></p>
      <div class="onboarding-actions">
        <button class="btn btn-ghost" data-tour-skip>Skip</button>
        <button class="btn btn-primary" data-tour-next>Next</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  const title = overlay.querySelector('h4');
  const body = overlay.querySelector('p');
  const nextBtn = overlay.querySelector('[data-tour-next]');
  const skipBtn = overlay.querySelector('[data-tour-skip]');
  let index = 0;

  function endTour() {
    overlay.remove();
    localStorage.setItem('coachTourDone', '1');
  }

  function renderStep() {
    const step = steps[index];
    if (!step) {
      endTour();
      return;
    }
    overlay.classList.add('active');
    title.textContent = step.title;
    body.textContent = step.body;
    nextBtn.textContent = index === steps.length - 1 ? 'Finish' : 'Next';
    const target = step.selector ? document.querySelector(step.selector) : null;
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  nextBtn?.addEventListener('click', () => {
    index += 1;
    if (index >= steps.length) {
      endTour();
    } else {
      renderStep();
    }
  });

  skipBtn?.addEventListener('click', () => endTour());
  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) {
      endTour();
    }
  });
  renderStep();
}

async function importCsv(form, endpoint, statusId) {
  const status = document.getElementById(statusId);
  const fileInput = form.querySelector('input[type="file"]');
  if (!fileInput || !fileInput.files.length) {
    if (status) status.textContent = 'Select a CSV file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  if (status) status.textContent = 'Uploading...';
  try {
    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Import failed.');
    }
    if (status) status.textContent = `Imported ${data.created || 0} rows. ${
      data.errors?.length ? data.errors.length + ' errors' : 'No errors.'
    }`;
    showToast(status.textContent);
  } catch (error) {
    if (status) status.textContent = error.message || 'Import failed.';
    showToast(error.message || 'Import failed.');
  }
}

function renderForecast(resultSet, profile) {
  const output = document.getElementById('forecastOutput');
  if (!output) {
    return;
  }
  if (!resultSet || !resultSet.length) {
    output.innerHTML = '<div class="muted">No forecast available yet.</div>';
    return;
  }
  const profileBlock = profile
    ? `
      <div class="forecast-profile">
        <h4>${profile.name}</h4>
        <p class="muted small">Height: ${profile.height_cm ?? '—'} cm · Weight: ${profile.weight_kg ?? '—'} kg · BMI: ${
        profile.bmi != null ? formatNumber(profile.bmi, 1) : '—'
      }</p>
        <p class="muted small">Bench 1RM: ${
          profile.bench_1rm_kg != null ? formatNumber(profile.bench_1rm_kg, 1) + ' kg' : '—'
        } · Squat 1RM: ${profile.squat_1rm_kg != null ? formatNumber(profile.squat_1rm_kg, 1) + ' kg' : '—'}</p>
      </div>
    `
    : '';

  const sections = resultSet
    .map(({ metric, forecast }) => {
      const { model, trend, confidence, forecasts } = forecast || {};
      const title = metric.replace('_', ' ').toUpperCase();
      const insight = describeForecast(metric, forecast);
      const summary = `
        <div class="forecast-summary">
          <span><strong>Model:</strong> ${model || 'n/a'}</span>
          <span><strong>Trend:</strong> ${trend || 'n/a'}</span>
          <span><strong>RMSE:</strong> ${confidence ?? 'n/a'}</span>
        </div>
        <p class="muted small">${insight}</p>
      `;
      const rows =
        (forecasts || [])
          .map(([date, value]) => `<tr><td>${date}</td><td>${formatNumber(value)}</td></tr>`)
          .join('') || '<tr><td colspan="2">Insufficient data.</td></tr>';
      return `
        <div class="forecast-section">
          <h5>${title}</h5>
          ${summary}
          <table class="forecast-table">
            <thead><tr><th>Day</th><th>Forecast</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    })
    .join('');
  output.innerHTML = profileBlock + sections;
}

function describeForecast(metric, forecast) {
  const points = (forecast?.forecasts || []).map(([, value]) => Number(value)).filter((v) => !Number.isNaN(v));
  if (!points.length) {
    return 'Not enough data to explain this forecast yet.';
  }
  const start = points[0];
  const end = points[points.length - 1];
  const change = end - start;
  const direction = change > 0 ? 'uptrend' : change < 0 ? 'downtrend' : 'flat trend';
  const pct = start !== 0 ? (change / Math.abs(start)) * 100 : 0;
  const magnitude =
    Math.abs(pct) < 5
      ? 'minor shift'
      : Math.abs(pct) < 15
      ? 'moderate change'
      : 'notable change';
  const metricLabel = metric.replace('_', ' ');
  const model = forecast?.model || 'n/a';
  const rmse = forecast?.confidence ?? 'n/a';
  const trendWord = direction === 'flat trend' ? 'stable pattern' : direction;
  const rationale = [
    `The model (${model}) projects a ${trendWord} for ${metricLabel} with a ${magnitude} (${formatNumber(change, 1)} absolute, ${formatNumber(
      pct,
      1
    )}% over the horizon).`,
    'This trajectory reflects recent volume and RPE trends: higher recent load typically sustains or lifts performance, while load reductions or elevated RPE with lower output can lead to plateaus or drops.',
    'We collapse multiple same-day logs and tame single outliers, so the slope reflects consistent behaviour rather than one-off spikes.',
    'Tags and notes influence the slope: consistent “technical” or “light” sessions often create gentler trends, while heavy/competition tags can steepen short-term changes.',
    `Model confidence (RMSE ${rmse}) indicates how tightly the forecast follows your history; lower RMSE means tighter fit, higher RMSE means greater uncertainty—use coaching judgment when adjusting plans.`,
    'If you see a downward bend, consider easing acute load and prioritizing quality throws; if upward, verify readiness to sustain intensity without overreaching.'
  ];
  return rationale.join(' ');
}
  const selectField = document.getElementById('athleteSelect');
  if (selectField) {
    selectField.innerHTML = state.athletes
      .map((athlete) => `<option value="${athlete.id}">${athlete.name}</option>`)
      .join('');
  }
