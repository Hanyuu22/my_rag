import { RotateCcw } from 'lucide-react'
import type { Settings } from '../hooks/useSettings'
import { DEFAULT_SETTINGS } from '../hooks/useSettings'

const USER_AVATARS = ['/avatar_user.png', '🙋', '👨‍💻', '👩‍💻', '🧑‍🔬', '🧑‍💼', '🤓', '🧑', '👦', '👧', '🧑‍🏫', '🧑‍🎓', '🕵️']
const BOT_AVATARS  = ['/avatar_bot.png', '/avatar_bot2.png', '🤖', '🧠', '🌿', '📚', '🔬', '💡', '⚡', '🦾', '🌟', '✨', '🍃', '🔭']

interface Props {
  settings: Settings
  onUpdate: (patch: Partial<Settings>) => void
  onReset: () => void
}

export default function SettingsPage({ settings, onUpdate, onReset }: Props) {
  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <div style={{ maxWidth: 860, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* 标题 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--green-deep)' }}>参数设置</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 3 }}>所有设置自动保存，刷新不丢失</div>
          </div>
          <button
            className="btn-ghost"
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={onReset}
          >
            <RotateCcw size={13} /> 恢复默认
          </button>
        </div>

        {/* 流程控制 */}
        <div className="glass-card" style={{ padding: 24 }}>
          <SectionTitle>流程控制</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>

            <PipelineToggle
              label="BGE Reranker 精排"
              hint="关闭后跳过精排，检索更快但精度下降"
              enabled={settings.rerankerEnabled}
              onChange={v => onUpdate({ rerankerEnabled: v })}
            />

            <PipelineToggle
              label="Fallback 兜底路由"
              hint="知识库无结果时自动走 LLM / 网络搜索"
              enabled={settings.fallbackEnabled}
              onChange={v => onUpdate({ fallbackEnabled: v })}
            >
              {settings.fallbackEnabled && (
                <div style={{ display: 'flex', gap: 5, marginTop: 8, flexWrap: 'wrap' }}>
                  {(['auto', 'llm', 'web'] as const).map(m => (
                    <button key={m} onClick={() => onUpdate({ fallbackMethod: m })} style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 10, cursor: 'pointer',
                      border: `1px solid ${settings.fallbackMethod === m ? 'var(--green-soft)' : 'rgba(100,180,80,0.2)'}`,
                      background: settings.fallbackMethod === m ? 'rgba(90,170,76,0.15)' : 'rgba(255,255,255,0.5)',
                      color: settings.fallbackMethod === m ? 'var(--green-mid)' : 'var(--text-dim)',
                      fontWeight: settings.fallbackMethod === m ? 600 : 400,
                    }}>
                      {{ auto: '自动', llm: 'LLM知识', web: '网络搜索' }[m]}
                    </button>
                  ))}
                </div>
              )}
            </PipelineToggle>

            <PipelineToggle
              label="知识图谱检索"
              hint="GraphRAG：实体关系图辅助检索"
              enabled={false}
              onChange={() => {}}
              comingSoon
            />

            <PipelineToggle
              label="RAPTOR 递归摘要"
              hint="多层摘要树，增强长文档理解"
              enabled={false}
              onChange={() => {}}
              comingSoon
            />

            <PipelineToggle
              label="多路召回融合"
              hint="Hybrid BM25 + Dense，关闭后仅用向量检索"
              enabled={true}
              onChange={() => {}}
              comingSoon
            />

          </div>
        </div>

        {/* 双列布局 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* 检索参数 */}
          <div className="glass-card" style={{ padding: 24 }}>
            <SectionTitle>检索参数</SectionTitle>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <SliderRow
                label="初检数量 (Top-K)"
                hint="BM25 和向量各召回多少条"
                value={settings.retrieverTopK} min={3} max={30} step={1}
                onChange={v => onUpdate({ retrieverTopK: v })}
                defaultVal={DEFAULT_SETTINGS.retrieverTopK}
              />
              <SliderRow
                label="精排保留 (Reranker Top-N)"
                hint="Reranker 后保留的最终文档数"
                value={settings.rerankerTopN} min={1} max={8} step={1}
                onChange={v => onUpdate({ rerankerTopN: v })}
                defaultVal={DEFAULT_SETTINGS.rerankerTopN}
              />
              <SliderRow
                label="改写阈值"
                hint="置信分低于此值时触发 Query Rewriting"
                value={settings.scoreThreshold} min={0.1} max={0.9} step={0.05}
                onChange={v => onUpdate({ scoreThreshold: v })}
                defaultVal={DEFAULT_SETTINGS.scoreThreshold}
                format={v => v.toFixed(2)}
              />
              <SliderRow
                label="最大改写次数"
                hint="改写重试上限，超过后强制生成"
                value={settings.maxRetry} min={0} max={4} step={1}
                onChange={v => onUpdate({ maxRetry: v })}
                defaultVal={DEFAULT_SETTINGS.maxRetry}
              />
            </div>
          </div>

          {/* 生成参数 + 头像 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="glass-card" style={{ padding: 24 }}>
              <SectionTitle>生成参数</SectionTitle>
              <SliderRow
                label="Temperature"
                hint="越高越有创意，越低越保守忠实原文"
                value={settings.temperature} min={0} max={1} step={0.05}
                onChange={v => onUpdate({ temperature: v })}
                defaultVal={DEFAULT_SETTINGS.temperature}
                format={v => v.toFixed(2)}
              />
            </div>

            <div className="glass-card" style={{ padding: 24 }}>
              <SectionTitle>头像设置</SectionTitle>

              {/* 当前预览 */}
              <div style={{ display: 'flex', gap: 20, marginBottom: 20 }}>
                <AvatarPreview src={settings.userAvatar} label="我" />
                <AvatarPreview src={settings.botAvatar} label="Bot" />
              </div>

              <div style={{ marginBottom: 8, fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)' }}>用户头像</div>
              <EmojiGrid options={USER_AVATARS} selected={settings.userAvatar} onSelect={v => onUpdate({ userAvatar: v })} />
              <div style={{ margin: '16px 0 8px', fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)' }}>机器人头像</div>
              <EmojiGrid options={BOT_AVATARS} selected={settings.botAvatar} onSelect={v => onUpdate({ botAvatar: v })} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── 子组件 ── */

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, color: 'var(--text-dim)',
      textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 18,
    }}>{children}</div>
  )
}

function AvatarPreview({ src, label }: { src: string; label: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 48, height: 48, borderRadius: '50%', overflow: 'hidden',
        border: '2px solid rgba(100,180,80,0.3)',
        background: 'rgba(255,255,255,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 26,
      }}>
        {src.startsWith('/')
          ? <img src={src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          : src}
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{label}</span>
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
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 500 }}>{label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {!isDefault && (
            <button onClick={() => onChange(defaultVal)}
              style={{ fontSize: 11, color: 'var(--text-dim)', background: 'none', border: 'none', cursor: 'pointer' }}
              title="恢复此项默认">↩</button>
          )}
          <span style={{
            fontSize: 14, fontWeight: 700, minWidth: 40, textAlign: 'right',
            color: isDefault ? 'var(--text-dim)' : 'var(--green-mid)',
            fontVariantNumeric: 'tabular-nums',
          }}>{display}</span>
        </div>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%', height: 5, appearance: 'none',
          background: `linear-gradient(to right, var(--green-soft) 0%, var(--green-soft) ${pct}%, rgba(100,180,80,0.15) ${pct}%, rgba(100,180,80,0.15) 100%)`,
          borderRadius: 999, outline: 'none', cursor: 'pointer', border: 'none', padding: 0,
        }}
      />
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>{hint}</div>
    </div>
  )
}

function PipelineToggle({ label, hint, enabled, onChange, comingSoon, children }: {
  label: string; hint: string
  enabled: boolean; onChange: (v: boolean) => void
  comingSoon?: boolean; children?: React.ReactNode
}) {
  return (
    <div style={{
      padding: '12px 14px', borderRadius: 12,
      background: comingSoon ? 'rgba(200,200,200,0.06)' : enabled ? 'rgba(90,170,76,0.07)' : 'rgba(255,255,255,0.5)',
      border: `1px solid ${comingSoon ? 'rgba(150,150,150,0.15)' : enabled ? 'rgba(90,170,76,0.25)' : 'rgba(150,150,150,0.2)'}`,
      opacity: comingSoon ? 0.6 : 1,
      transition: 'all 0.2s',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: comingSoon ? 'var(--text-dim)' : 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 5 }}>
            {label}
            {comingSoon && <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 4, background: 'rgba(150,150,150,0.15)', color: 'var(--text-dim)', fontWeight: 500 }}>开发中</span>}
          </div>
          <div style={{ fontSize: 10.5, color: 'var(--text-dim)', marginTop: 2 }}>{hint}</div>
        </div>
        {/* Toggle 开关 */}
        <div
          onClick={() => !comingSoon && onChange(!enabled)}
          style={{
            width: 36, height: 20, borderRadius: 10, flexShrink: 0,
            background: enabled && !comingSoon ? 'var(--green-soft)' : 'rgba(150,150,150,0.25)',
            position: 'relative', cursor: comingSoon ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s',
          }}
        >
          <div style={{
            position: 'absolute', top: 3, width: 14, height: 14, borderRadius: '50%',
            background: 'white', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
            left: enabled && !comingSoon ? 19 : 3,
            transition: 'left 0.2s',
          }} />
        </div>
      </div>
      {children}
    </div>
  )
}

function EmojiGrid({ options, selected, onSelect }: { options: string[]; selected: string; onSelect: (v: string) => void }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
      {options.map(opt => (
        <button key={opt} onClick={() => onSelect(opt)} style={{
          width: 38, height: 38, fontSize: 18, borderRadius: 9, cursor: 'pointer',
          border: selected === opt ? '2px solid var(--green-soft)' : '1.5px solid rgba(100,180,80,0.2)',
          background: selected === opt ? 'rgba(114,184,100,0.15)' : 'rgba(255,255,255,0.5)',
          transition: 'all 0.15s', display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: selected === opt ? '0 0 0 3px rgba(114,184,100,0.12)' : 'none',
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
