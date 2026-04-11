import { useState } from 'react'
import Sidebar, { type Page } from './components/Sidebar'
import AboutPage from './pages/AboutPage'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import SettingsPage from './pages/SettingsPage'
import { useSettings } from './hooks/useSettings'
import './index.css'

export default function App() {
  const [page, setPage] = useState<Page>('chat')
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const { settings, update, reset } = useSettings()

  return (
    <div style={{ height: '100vh', display: 'flex', overflow: 'hidden' }}>
      {/* 左侧导航 */}
      <Sidebar current={page} onChange={setPage} />

      {/* 主内容区 */}
      <div style={{ flex: 1, padding: 20, overflow: 'hidden', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {page === 'about' && <AboutPage />}
        {page === 'chat' && (
          <ChatPage settings={settings} refreshTrigger={refreshTrigger} />
        )}
        {page === 'knowledge' && (
          <KnowledgePage
            refreshTrigger={refreshTrigger}
            onUploaded={() => setRefreshTrigger(n => n + 1)}
          />
        )}
        {page === 'settings' && (
          <SettingsPage settings={settings} onUpdate={update} onReset={reset} />
        )}
      </div>

      {/* 右下角水印 */}
      <div style={{
        position: 'fixed', bottom: 14, right: 18,
        fontSize: 10.5, color: 'var(--text-dim)',
        opacity: 0.45, transition: 'opacity 0.2s',
        pointerEvents: 'auto', userSelect: 'none',
        display: 'flex', alignItems: 'center', gap: 5,
        fontFamily: "'PingFang SC', 'Microsoft YaHei', sans-serif",
        letterSpacing: '0.03em',
      }}
        onMouseEnter={e => (e.currentTarget.style.opacity = '0.85')}
        onMouseLeave={e => (e.currentTarget.style.opacity = '0.45')}
      >
        <span style={{ pointerEvents: 'none' }}>作者：Hanyuu（朱添乐）</span>
        <span style={{ color: 'var(--green-light)', pointerEvents: 'none' }}>×</span>
        <span style={{ pointerEvents: 'none' }}>Claude Sonnet 4.6</span>
      </div>
    </div>
  )
}
