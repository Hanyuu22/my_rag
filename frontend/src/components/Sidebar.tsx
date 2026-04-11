import { MessageSquare, BookOpen, Settings } from 'lucide-react'

export type Page = 'about' | 'chat' | 'knowledge' | 'settings'

interface Props {
  current: Page
  onChange: (page: Page) => void
}

const ITEMS = [
  { id: 'chat',      icon: MessageSquare, label: '问答'  },
  { id: 'knowledge', icon: BookOpen,      label: '知识库' },
  { id: 'settings',  icon: Settings,      label: '设置'  },
] as const

export default function Sidebar({ current, onChange }: Props) {
  return (
    <div style={{
      width: 72, flexShrink: 0, height: '100%',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      paddingTop: 18, paddingBottom: 18, gap: 4,
      background: 'rgba(255,255,255,0.55)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid rgba(100,180,80,0.15)',
    }}>
      {/* Logo — 点击进入关于页 */}
      <div
        onClick={() => onChange('about')}
        title="关于本系统"
        style={{
          marginBottom: 16, width: 58, height: 58, cursor: 'pointer',
          transition: 'transform 0.18s, box-shadow 0.18s',
          borderRadius: 14,
          boxShadow: current === 'about'
            ? '0 0 0 2.5px var(--green-soft), 0 4px 16px rgba(74,140,63,0.35)'
            : '0 3px 12px rgba(74,140,63,0.28), 0 1px 4px rgba(0,0,0,0.1)',
          transform: current === 'about' ? 'scale(1.05)' : 'scale(1)',
        }}
        onMouseEnter={e => {
          if (current !== 'about') (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.07)'
        }}
        onMouseLeave={e => {
          if (current !== 'about') (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)'
        }}
      >
        <img
          src="/logo.png"
          alt="logo"
          style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 14 }}
        />
      </div>

      {ITEMS.map(({ id, icon: Icon, label }) => {
        const active = current === id
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            title={label}
            style={{
              width: 52, height: 56, borderRadius: 14, border: 'none',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', gap: 4, cursor: 'pointer', transition: 'all 0.2s',
              background: active ? 'rgba(90,170,76,0.15)' : 'transparent',
              color: active ? 'var(--green-mid)' : 'var(--text-dim)',
              boxShadow: active ? 'inset 0 0 0 1.5px rgba(90,170,76,0.3)' : 'none',
            }}
          >
            <Icon size={20} strokeWidth={active ? 2.2 : 1.8} />
            <span style={{ fontSize: 10, fontWeight: active ? 600 : 400, letterSpacing: '0.02em' }}>
              {label}
            </span>
          </button>
        )
      })}
    </div>
  )
}
