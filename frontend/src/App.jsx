import { useEffect, useState } from 'react'
import './App.css'

// ---------------------------------------------------------------
// 1. DATA THAT NEVER CHANGES
// The topic cards the student can tap. "value" is what we send to the backend.
// ---------------------------------------------------------------
const TOPICS = [
  { emoji: '📚', label: 'Exams & deadlines', value: 'Exams and deadlines' },
  { emoji: '🍎', label: 'Food', value: 'Food' },
  { emoji: '🏠', label: 'Housing', value: 'Housing' },
  { emoji: '🔑', label: 'Locked out', value: 'Locked out' },
  { emoji: '💬', label: 'Feeling down', value: 'Feeling down' },
  { emoji: '👋', label: 'Lonely', value: 'Lonely' },
]

// The steps shown on the "agents at work" screen, in order
const AGENTS = [
  { name: 'Safety', doing: 'Checking for anything urgent' },
  { name: 'Intake', doing: 'Reading what you told us' },
  { name: 'Schedule', doing: 'Building your plan' },
  { name: 'Resources', doing: 'Matching UNC resources' },
]

// ---------------------------------------------------------------
// 2. THE MAIN COMPONENT
// A component is a function that returns what should be on the screen.
// ---------------------------------------------------------------
export default function App() {
  // STATE: data that can change. When it changes, React redraws the screen.
  // useState(startingValue) gives back [currentValue, functionToChangeIt].
  const [message, setMessage] = useState('')
  const [topics, setTopics] = useState([])
  const [deadlines, setDeadlines] = useState([{ title: '', date: '' }])
  const [loading, setLoading] = useState(false)
  const [activeStep, setActiveStep] = useState(0)
  const [result, setResult] = useState(() => loadSaved(SAVED_PLAN_KEY, null))
  const [error, setError] = useState('')
  const [editing, setEditing] = useState(false) // true = showing the form again over an existing plan

  // Save the latest plan whenever it changes
  useEffect(() => {
    if (result) save(SAVED_PLAN_KEY, result)
  }, [result])

  // While loading, move the "agents at work" highlight forward every 2.5 seconds
  useEffect(() => {
    if (!loading) return
    setActiveStep(0)
    const timer = setInterval(() => {
      setActiveStep((step) => Math.min(step + 1, AGENTS.length - 1))
    }, 2500)
    return () => clearInterval(timer) // stop the timer when loading ends
  }, [loading])

  // Tap a topic card: add it if it's not picked, remove it if it is
  function toggleTopic(value) {
    setTopics((current) =>
      current.includes(value) ? current.filter((t) => t !== value) : [...current, value]
    )
  }

  // Change one field ("title" or "date") of one deadline row
  function updateDeadline(index, field, value) {
    setDeadlines((current) =>
      current.map((d, i) => (i === index ? { ...d, [field]: value } : d))
    )
  }

  function addDeadline() {
    setDeadlines((current) => [...current, { title: '', date: '' }])
  }

  function removeDeadline(index) {
    setDeadlines((current) => current.filter((_, i) => i !== index))
  }

  // Send the check-in to our Python backend (through the Vite proxy)
  async function buildPlan(event) {
    event.preventDefault() // stop the browser from reloading the page
    setError('')
    setLoading(true)
    try {
      const response = await fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          topics,
          deadlines: deadlines.filter((d) => d.title.trim() && d.date), // only filled-in rows
        }),
      })
      if (!response.ok) throw new Error(`Server answered ${response.status}`)
      setResult(await response.json())
      setEditing(false) // new plan is ready: switch to the tracker page
      window.scrollTo(0, 0)
    } catch (err) {
      setError("We couldn't reach the planner. Make sure the backend (Terminal 1) is running, then try again.")
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = !loading && (message.trim() || topics.length > 0 || deadlines.some((d) => d.title && d.date))

  // Which page to show: the check-in form, or the plan (tracker) page
  const showForm = !loading && (!result || editing)
  const showPlan = !loading && result && !editing

  // ---------------------------------------------------------------
  // 3. WHAT SHOWS ON THE SCREEN (JSX: HTML-like code inside JavaScript)
  // ---------------------------------------------------------------
  return (
    <div className="app">
      {/* App name: always visible at the top */}
      <nav className="topbar">
        <span className="brand">
          <span className="brand-icon" aria-hidden="true">🌱</span> Estelle Supporter
        </span>
        <span className="tagline">UNC student support</span>
      </nav>

      {showForm && (
      <>
      {result && (
        <div className="header">
          <button type="button" className="link-button" onClick={() => setEditing(false)}>
            ← Back to my tracker
          </button>
        </div>
      )}
      <header className="header">
        <h1>Let's make a plan</h1>
        <p className="subtitle">
          Tell us what's going on. We'll check for anything urgent, then turn it into steps for today,
          this week, and this month.
        </p>
      </header>

      <form className="card" onSubmit={buildPlan}>
        {/* Topic cards */}
        <h2>What's going on?</h2>
        <p className="hint">Tap everything that fits.</p>
        <div className="topics">
          {TOPICS.map((topic) => (
            <button
              type="button"
              key={topic.value}
              className={topics.includes(topic.value) ? 'topic selected' : 'topic'}
              onClick={() => toggleTopic(topic.value)}
              aria-pressed={topics.includes(topic.value)}
            >
              <span className="emoji">{topic.emoji}</span>
              {topic.label}
            </button>
          ))}
        </div>

        {/* Free-text box */}
        <label className="field-label" htmlFor="message">Tell us more</label>
        <textarea
          id="message"
          rows={4}
          placeholder="Example: I have a chem midterm Thursday and an essay due Friday, and I'm really stressed."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />

        {/* Deadlines with date pickers */}
        <h2>Upcoming deadlines</h2>
        <p className="hint">Optional. Adding a date puts it on your calendar.</p>
        {deadlines.map((d, index) => (
          <div className="deadline-row" key={index}>
            <input
              type="text"
              placeholder="What's due? (e.g. Chem midterm)"
              value={d.title}
              onChange={(e) => updateDeadline(index, 'title', e.target.value)}
            />
            <input
              type="date"
              value={d.date}
              onChange={(e) => updateDeadline(index, 'date', e.target.value)}
            />
            {deadlines.length > 1 && (
              <button type="button" className="remove" onClick={() => removeDeadline(index)} aria-label="Remove deadline">
                ✕
              </button>
            )}
          </div>
        ))}
        <button type="button" className="link-button" onClick={addDeadline}>+ Add another deadline</button>

        <button type="submit" className="primary" disabled={!canSubmit}>
          {loading ? 'Building your plan…' : 'Build my plan →'}
        </button>
        {error && <p className="error">{error}</p>}
      </form>
      </>
      )}

      {/* "Agents at work" while waiting */}
      {loading && (
        <section className="card">
          <h2>Our agents are working on it</h2>
          <ul className="agents">
            {AGENTS.map((agent, i) => (
              <li key={agent.name} className={i < activeStep ? 'done' : i === activeStep ? 'active' : 'waiting'}>
                <span className="dot" />
                <strong>{agent.name}</strong> · {agent.doing}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* The plan page: replaces the form once the plan is ready */}
      {showPlan && (
        <PlanPage
          result={result}
          onNewCheckIn={() => {
            setEditing(true)
            window.scrollTo(0, 0)
          }}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------
// 4. THE PLAN PAGE: what the student sees after clicking "Build my plan"
// Order: life-safety help (only if needed) -> tracker -> support card -> agent log
// ---------------------------------------------------------------
const LIFE_SAFETY_GROUPS = ['crisis', 'active_threat', 'danger'] // these must always come first

function PlanPage({ result, onNewCheckIn }) {
  const urgentGroups = result.urgent.groups.filter((g) => LIFE_SAFETY_GROUPS.includes(g.group))
  const supportGroups = result.urgent.groups.filter((g) => !LIFE_SAFETY_GROUPS.includes(g.group))
  const hasPlan = result.tasks.length > 0 || result.deadlines.length > 0
  const hasSupport = supportGroups.length > 0 || result.resources.length > 0

  return (
    <>
      <div className="plan-bar">
        <h1>Your plan</h1>
        <button type="button" className="secondary" onClick={onNewCheckIn}>
          + New check-in
        </button>
      </div>

      {/* Life-safety help from the Safety agent: always on top, in red */}
      {urgentGroups.map((group) => (
        <section className="card urgent" key={group.group}>
          <p className="urgent-intro">{group.intro}</p>
          <ResourceList resources={group.resources} />
          {group.outro && <p className="urgent-outro">{group.outro}</p>}
        </section>
      ))}

      {/* The main thing: Today / Week / Month */}
      {hasPlan && <Tracker result={result} />}

      {/* Food, housing, feeling down, lonely, lockouts + matched UNC resources */}
      {hasSupport && (
        <section className="card support">
          <h2>Support for you</h2>
          {supportGroups.map((group) => (
            <div className="support-group" key={group.group}>
              <p className="support-intro">{group.intro}</p>
              <ResourceList resources={group.resources} />
              {group.outro && <p className="hint">{group.outro}</p>}
            </div>
          ))}
          {result.resources.length > 0 && (
            <div className="support-group">
              {supportGroups.length > 0 && <p className="support-intro">More UNC resources that match what you shared:</p>}
              <ResourceList resources={result.resources} />
            </div>
          )}
        </section>
      )}

      {!hasPlan && !hasSupport && urgentGroups.length === 0 && (
        <section className="card">
          <p className="hint">
            We didn't find anything to plan yet. Try adding a deadline with a date, or tell us a little more.
          </p>
        </section>
      )}

      {result.notes.map((note) => (
        <p className="hint note" key={note}>{note}</p>
      ))}

      {/* Small, folded-up log of what each agent did (click to open) */}
      <details className="agent-log">
        <summary>What our agents did</summary>
        <ul className="agents">
          {result.steps.map((step) => (
            <li key={step.agent} className={step.status === 'done' ? 'done' : 'waiting'}>
              <span className="dot" />
              <strong>{step.agent}</strong> · {step.status}
            </li>
          ))}
        </ul>
      </details>
    </>
  )
}

// Shows a list of resources with clickable phone numbers and links
function ResourceList({ resources }) {
  return (
    <ul className="resources">
      {resources.map((r) => (
        <li key={r.name}>
          <strong>{r.name}</strong>
          {r.phone && <> · <a href={`tel:${r.phone}`}>{r.phone}</a></>}
          {r.details && <span className="details"> ({r.details})</span>}
          {r.link && (
            <div><a href={r.link} target="_blank" rel="noreferrer">{r.link}</a></div>
          )}
        </li>
      ))}
    </ul>
  )
}

// ---------------------------------------------------------------
// 5. SAVING IN THE BROWSER (localStorage)
// The browser keeps small bits of data even after a refresh.
// try/catch: if the browser blocks storage, the app still works.
// ---------------------------------------------------------------
const SAVED_PLAN_KEY = 'estelle-plan'
const SAVED_DONE_KEY = 'estelle-done'

function loadSaved(key, fallback) {
  try {
    const text = localStorage.getItem(key)
    return text ? JSON.parse(text) : fallback
  } catch {
    return fallback
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // storage not available: nothing to do
  }
}

// ---------------------------------------------------------------
// 6. DATE HELPERS
// Dates travel as text like "2026-10-23". These turn text <-> real dates.
// ---------------------------------------------------------------
function toKey(date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

function fromKey(key) {
  const [y, m, d] = key.split('-').map(Number)
  return new Date(y, m - 1, d) // local date, no time-zone surprises
}

function addDays(date, days) {
  const copy = new Date(date)
  copy.setDate(copy.getDate() + days)
  return copy
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

// ---------------------------------------------------------------
// 7. THE TRACKER
// ---------------------------------------------------------------
function Tracker({ result }) {
  const [view, setView] = useState('today') // which tab is open
  const [done, setDone] = useState(() => loadSaved(SAVED_DONE_KEY, {})) // { taskId: true }
  const todayKey = toKey(new Date())

  // Save checkmarks whenever they change
  useEffect(() => {
    save(SAVED_DONE_KEY, done)
  }, [done])

  function toggleDone(id) {
    setDone((current) => ({ ...current, [id]: !current[id] }))
  }

  // Group work blocks and deadlines by date: { "2026-10-23": [...] }
  const tasksByDate = {}
  for (const task of result.tasks) {
    ;(tasksByDate[task.date] ||= []).push(task)
  }
  const deadlinesByDate = {}
  for (const d of result.deadlines) {
    ;(deadlinesByDate[d.date] ||= []).push(d)
  }

  // Progress for the next 7 days (breaks don't count)
  const weekKeys = Array.from({ length: 7 }, (_, i) => toKey(addDays(fromKey(todayKey), i)))
  const weekWork = weekKeys.flatMap((k) => (tasksByDate[k] || []).filter((t) => !t.is_break))
  const weekDone = weekWork.filter((t) => done[t.id]).length

  const shared = { tasksByDate, deadlinesByDate, done, toggleDone, todayKey }

  return (
    <section className="card tracker">
      <div className="tracker-head">
        <h2>Your tracker</h2>
        <div className="tabs" role="tablist">
          {['today', 'week', 'month'].map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={view === name}
              className={view === name ? 'tab active' : 'tab'}
              onClick={() => setView(name)}
            >
              {name[0].toUpperCase() + name.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {weekWork.length > 0 && (
        <div className="progress">
          <div className="progress-label">
            {weekDone} of {weekWork.length} done this week
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${(weekDone / weekWork.length) * 100}%` }} />
          </div>
        </div>
      )}

      {view === 'today' && <TodayView {...shared} />}
      {view === 'week' && <WeekView {...shared} weekKeys={weekKeys} />}
      {view === 'month' && <MonthView {...shared} />}
    </section>
  )
}

// One task with a checkbox (breaks have no checkbox)
function TaskItem({ task, done, toggleDone, showTime = true }) {
  if (task.is_break) {
    return <li className="task break">{showTime && <span className="time">{task.time_label}</span>}Break</li>
  }
  return (
    <li className={done[task.id] ? 'task done' : 'task'}>
      <label>
        <input type="checkbox" checked={!!done[task.id]} onChange={() => toggleDone(task.id)} />
        {showTime && <span className="time">{task.time_label}</span>}
        <span className="task-title">{task.title}</span>
        {task.duration_label && <span className="duration">{task.duration_label}</span>}
      </label>
    </li>
  )
}

function DeadlineList({ items }) {
  if (!items || items.length === 0) return null
  return (
    <ul className="deadline-list">
      {items.map((d) => (
        <li key={d.title + d.date}>🔴 Due: {d.title}</li>
      ))}
    </ul>
  )
}

// TODAY: today's timeline. If today is empty, show the next day that has something.
function TodayView({ tasksByDate, deadlinesByDate, done, toggleDone, todayKey }) {
  const todayTasks = tasksByDate[todayKey] || []
  const nextKey = Object.keys(tasksByDate).sort().find((k) => k > todayKey)

  return (
    <div>
      <p className="view-date">{fromKey(todayKey).toDateString()}</p>
      <DeadlineList items={deadlinesByDate[todayKey]} />
      {todayTasks.length > 0 ? (
        <ul className="task-list">
          {todayTasks.map((task) => (
            <TaskItem key={task.id} task={task} done={done} toggleDone={toggleDone} />
          ))}
        </ul>
      ) : (
        <p className="hint">
          Nothing planned for today.
          {nextKey && ` Your next block is on ${fromKey(nextKey).toDateString()}. Check the Week tab.`}
        </p>
      )}
    </div>
  )
}

// WEEK: the next 7 days, one column each
function WeekView({ tasksByDate, deadlinesByDate, done, toggleDone, todayKey, weekKeys }) {
  return (
    <div className="week">
      {weekKeys.map((key) => {
        const date = fromKey(key)
        const work = (tasksByDate[key] || []).filter((t) => !t.is_break)
        return (
          <div key={key} className={key === todayKey ? 'day today' : 'day'}>
            <div className="day-name">
              {DAY_NAMES[date.getDay()]} <span>{date.getMonth() + 1}/{date.getDate()}</span>
            </div>
            <DeadlineList items={deadlinesByDate[key]} />
            {work.length === 0 ? (
              <p className="empty">Free</p>
            ) : (
              <ul className="task-list compact">
                {work.map((task) => (
                  <TaskItem key={task.id} task={task} done={done} toggleDone={toggleDone} />
                ))}
              </ul>
            )}
          </div>
        )
      })}
    </div>
  )
}

// MONTH: a calendar grid. Tap a day to see its tasks underneath.
function MonthView({ tasksByDate, deadlinesByDate, done, toggleDone, todayKey }) {
  const [monthStart, setMonthStart] = useState(() => {
    const t = fromKey(todayKey)
    return new Date(t.getFullYear(), t.getMonth(), 1)
  })
  const [selected, setSelected] = useState(todayKey)

  // Build the grid: blank cells before the 1st, then every day of the month
  const blanks = monthStart.getDay()
  const daysInMonth = new Date(monthStart.getFullYear(), monthStart.getMonth() + 1, 0).getDate()
  const cells = [
    ...Array.from({ length: blanks }, () => null),
    ...Array.from({ length: daysInMonth }, (_, i) => toKey(new Date(monthStart.getFullYear(), monthStart.getMonth(), i + 1))),
  ]

  function changeMonth(step) {
    setMonthStart(new Date(monthStart.getFullYear(), monthStart.getMonth() + step, 1))
  }

  const selectedWork = (tasksByDate[selected] || []).filter((t) => !t.is_break)

  return (
    <div>
      <div className="month-head">
        <button type="button" className="link-button" onClick={() => changeMonth(-1)} aria-label="Previous month">‹</button>
        <strong>{monthStart.toLocaleString('default', { month: 'long', year: 'numeric' })}</strong>
        <button type="button" className="link-button" onClick={() => changeMonth(1)} aria-label="Next month">›</button>
      </div>

      <div className="month-grid">
        {DAY_NAMES.map((name) => <div key={name} className="weekday">{name}</div>)}
        {cells.map((key, i) => {
          if (!key) return <div key={`blank-${i}`} />
          const work = (tasksByDate[key] || []).filter((t) => !t.is_break)
          const allDone = work.length > 0 && work.every((t) => done[t.id])
          const classes = ['cell']
          if (key === todayKey) classes.push('today')
          if (key === selected) classes.push('selected')
          return (
            <button type="button" key={key} className={classes.join(' ')} onClick={() => setSelected(key)}>
              <span className="cell-num">{fromKey(key).getDate()}</span>
              {deadlinesByDate[key] && <span className="cell-deadline" title="Deadline">🔴</span>}
              {work.length > 0 && <span className={allDone ? 'cell-count all-done' : 'cell-count'}>{work.length}</span>}
            </button>
          )
        })}
      </div>

      <div className="month-detail">
        <p className="view-date">{fromKey(selected).toDateString()}</p>
        <DeadlineList items={deadlinesByDate[selected]} />
        {selectedWork.length > 0 ? (
          <ul className="task-list">
            {selectedWork.map((task) => (
              <TaskItem key={task.id} task={task} done={done} toggleDone={toggleDone} />
            ))}
          </ul>
        ) : (
          !deadlinesByDate[selected] && <p className="hint">Nothing planned this day.</p>
        )}
      </div>
    </div>
  )
}
