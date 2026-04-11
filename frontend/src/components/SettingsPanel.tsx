import { RotateCcw, X } from 'lucide-react'
import type { Settings } from '../hooks/useSettings'
import { DEFAULT_SETTINGS } from '../hooks/useSettings'

const USER_AVATARS = ['/avatar_user.png', '🙋', '👨‍💻', '👩‍💻', '🧑‍🔬', '🧑‍💼', '🤓', '🧑', '👦', '👧', '🧑‍🏫', '🧑‍🎓', '🕵️']
const BOT_AVATARS  = ['/avatar_bot.png', '/avatar_bot2.png', '🤖', '🧠', '🌿', '📚', '🔬', '💡', '⚡', '🦾', '🌟', '✨', '🍃', '🔭']

interface Props {
  settings: Settings
  onUpdate: (patch: Partial<Settings>) => void
  onReset: () => void
  onClose: () => void
}

export default function SettingsPanel({ settings, onUpdate, onReset, onClose }: Props) {
  return (
    <>
      {/* 遮罩 */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 200,
          background: 'rgba(200,230,195,0.3)',
          backdropFilter: 'blur(3px)',
        }}
      />

      {/* 抽屉 */}
      <div style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, width: 320,
        zIndex: 201, padding: 24, overflowY: 'auto',
        background: 'rgba(240,250,236,0.92)',
        backdropFilter: 'blur(20px)',
        borderLeft: '1px solid rgba(100,180,80,0.2)',
        boxShadow: '-8px 0 32px rgba(50,120,40,0.1)',
        display: 'flex', flexDirection: 'column', gap: 24,
      }}>
        {/* 标题 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--green-deep)' }}>⚙️ 参数设置</span>
          <button className="btn-ghost" style={{ padding: '4px 8px' }} onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        {/* 检索参数 */}
        <Section title="检索参数">
          <SliderRow
            label="初检数量 (Top-K)"
            hint="BM25 和向量各召回多少条"
            value={settings.retrieverTopK}
            min={3} max={30} step={1}
            onChange={v => onUpdate({ retrieverTopK: v })}
            defaultVal={DEFAULT_SETTINGS.retrieverTopK}
          />
          <SliderRow
            label="精排保留 (Reranker Top-N)"
            hint="Reranker 后保留的最终文档数"
            value={settings.rerankerTopN}
            min={1} max={8} step={1}
            onChange={v => onUpdate({ rerankerTopN: v })}
            defaultVal={DEFAULT_SETTINGS.rerankerTopN}
          />
          <SliderRow
            label="改写阈值"
            hint="置信分低于此值时触发 Query Rewriting"
            value={settings.scoreThreshold}
            min={0.1} max={0.9} step={0.05}
            onChange={v => onUpdate({ scoreThreshold: v })}
            defaultVal={DEFAULT_SETTINGS.scoreThreshold}
            format={v => v.toFixed(2)}
          />
          <SliderRow
            label="最大改写次数"
            hint="改写重试上限，超过后强制生成"
            value={settings.maxRetry}
            min={0} max={4} step={1}
            onChange={v => onUpdate({ maxRetry: v })}
            defaultVal={DEFAULT_SETTINGS.maxRetry}
          />
        </Section>

        {/* 生成参数 */}
        <Section title="生成参数">
          <SliderRow
            label="Temperature"
            hint="越高越有创意，越低越保守"
            value={settings.temperature}
            min={0} max={1} step={0.05}
            onChange={v => onUpdate({ temperature: v })}
            defaultVal={DEFAULT_SETTINGS.temperature}
            format={v => v.toFixed(2)}
          />
        </Section>

        {/* 头像设置 */}
        <Section title="头像设置">
          <div style={{ marginBottom: 4, fontSize: 12, color: 'var(--text-secondary)' }}>用户头像</div>
          <EmojiGrid
            options={USER_AVATARS}
            selected={settings.userAvatar}
            onSelect={v => onUpdate({ userAvatar: v })}
          />
          <div style={{ margin: '14px 0 4px', fontSize: 12, color: 'var(--text-secondary)' }}>机器人头像</div>
          <EmojiGrid
            options={BOT_AVATARS}
            selected={settings.botAvatar}
            onSelect={v => onUpdate({ botAvatar: v })}
          />
        </Section>

        {/* 恢复默认 */}
        <button
          className="btn-ghost"
          style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center', marginTop: 'auto' }}
          onClick={() => { onReset(); onClose() }}
        >
          <RotateCcw size={13} /> 恢复默认设置
        </button>
      </div>
    </>
  )
}

/* ── 子组件 ────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 11, fontWeight: 600, color: 'var(--text-dim)',
        textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 14,
      }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {children}
      </div>
    </div>
  )
}

interface SliderRowProps {
  label: string; hint: string
  value: number; min: number; max: number; step: number
  onChange: (v: number) => void
  defaultVal: number
  format?: (v: number) => string
}

function SliderRow({ label, hint, value, min, max, step, onChange, defaultVal, format }: SliderRowProps) {
  const display = format ? format(value) : String(value)
  const isDefault = Math.abs(value - defaultVal) < step * 0.1

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 500 }}>{label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {!isDefault && (
            <button
              onClick={() => onChange(defaultVal)}
              style={{ fontSize: 10, color: 'var(--text-dim)', background: 'none', border: 'none', cursor: 'pointer' }}
              title="恢复此项默认">↩</button>
          )}
          <span style={{
            fontSize: 13, fontWeight: 600, minWidth: 36, textAlign: 'right',
            color: isDefault ? 'var(--text-dim)' : 'var(--green-mid)',
            fontVariantNumeric: 'tabular-nums',
          }}>{display}</span>
        </div>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%', height: 4, appearance: 'none',
          background: `linear-gradient(to right, var(--green-soft) 0%, var(--green-soft) ${((value - min) / (max - min)) * 100}%, rgba(100,180,80,0.2) ${((value - min) / (max - min)) * 100}%, rgba(100,180,80,0.2) 100%)`,
          borderRadius: 999, outline: 'none', cursor: 'pointer', border: 'none', padding: 0,
        }}
      />
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 3 }}>{hint}</div>
    </div>
  )
}

function EmojiGrid({ options, selected, onSelect }: { options: string[]; selected: string; onSelect: (v: string) => void }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {options.map(opt => (
        <button
          key={opt}
          onClick={() => onSelect(opt)}
          style={{
            width: 36, height: 36, fontSize: 18, borderRadius: 8, cursor: 'pointer',
            border: selected === opt ? '2px solid var(--green-soft)' : '1.5px solid rgba(100,180,80,0.2)',
            background: selected === opt ? 'rgba(114,184,100,0.15)' : 'rgba(255,255,255,0.5)',
            transition: 'all 0.15s', display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: selected === opt ? '0 0 0 3px rgba(114,184,100,0.15)' : 'none',
            overflow: 'hidden', padding: 0,
          }}>
          {opt.startsWith('/')
            ? <img src={opt} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            : opt}
        </button>
      ))}
    </div>
  )
}
