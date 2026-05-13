import { useState, useRef, useCallback } from 'react'
import { Upload, FileText, X, CheckCircle, AlertCircle } from 'lucide-react'

interface TaskState {
  taskId: string
  filename: string
  collection: string
  status: 'pending' | 'processing' | 'done' | 'error'
  step: string
  percent: number
  error?: string
  chunks?: number
}

interface UploadZoneProps {
  onUploaded: () => void
}

const EMBEDDING_OPTIONS = [
  { value: 'BAAI/bge-large-zh-v1.5', label: 'bge-zh（中文）', hint: '适合纯中文文档' },
  { value: 'BAAI/bge-m3',            label: 'bge-m3（中英双语）', hint: '适合英文或中英混合文档' },
]

export default function UploadZone({ onUploaded }: UploadZoneProps) {
  const [dragging, setDragging]             = useState(false)
  const [showModal, setShowModal]           = useState(false)
  const [pendingFiles, setPendingFiles]     = useState<File[]>([])
  const [collectionName, setCollectionName] = useState('')
  const [useVlmOcr, setUseVlmOcr]           = useState(false)
  const [useParentChild, setUseParentChild] = useState(false)
  const [embeddingModel, setEmbeddingModel] = useState('BAAI/bge-large-zh-v1.5')
  const [tasks, setTasks]                   = useState<TaskState[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollingRefs  = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  const openModal = (files: File[]) => {
    if (files.length === 0) return
    setPendingFiles(files)
    // 用第一个文件名做默认 collection 名
    const defaultName = files[0].name.replace(/\.[^.]+$/, '').replace(/\s+/g, '_')
    setCollectionName(defaultName)
    setShowModal(true)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) openModal(files)
  }, [])

  const startPolling = (taskId: string) => {
    pollingRefs.current[taskId] = setInterval(async () => {
      try {
        const res  = await fetch(`/api/tasks/${taskId}`)
        const data = await res.json()
        setTasks(prev => prev.map(t =>
          t.taskId === taskId
            ? { ...t, status: data.status, step: data.step, percent: data.percent,
                error: data.error, chunks: data.result?.chunks }
            : t
        ))
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(pollingRefs.current[taskId])
          delete pollingRefs.current[taskId]
          if (data.status === 'done') onUploaded()
        }
      } catch {}
    }, 1000)
  }

  const handleUpload = async () => {
    if (pendingFiles.length === 0 || !collectionName.trim()) return
    setShowModal(false)

    // 多文件：每个文件独立提交，共享同一个 collection 和模型设置
    for (const file of pendingFiles) {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('collection_name', collectionName.trim())
      formData.append('use_vlm_ocr', String(useVlmOcr))
      formData.append('embedding_model', embeddingModel)
      formData.append('use_parent_child', String(useParentChild))

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData })
        if (!res.ok) {
          const err = await res.json()
          alert(`上传失败（${file.name}）：${err.detail}`)
          continue
        }
        const { task_id } = await res.json()
        const newTask: TaskState = {
          taskId: task_id, filename: file.name,
          collection: collectionName.trim(),
          status: 'pending', step: '等待处理...', percent: 0,
        }
        setTasks(prev => [newTask, ...prev])
        startPolling(task_id)
      } catch (e) {
        alert(`上传出错（${file.name}）：${e}`)
      }
    }

    setPendingFiles([])
    setCollectionName('')
  }

  const removeTask = (taskId: string) => {
    clearInterval(pollingRefs.current[taskId])
    delete pollingRefs.current[taskId]
    setTasks(prev => prev.filter(t => t.taskId !== taskId))
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* 拖拽上传区 */}
      <div
        className="glass-card"
        style={{
          padding: 32, textAlign: 'center', cursor: 'pointer',
          border: dragging ? '1.5px dashed var(--green-soft)' : '1.5px dashed rgba(100,180,80,0.3)',
          background: dragging ? 'rgba(114,184,100,0.1)' : 'rgba(255,255,255,0.4)',
          transition: 'all 0.2s',
        }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload size={32} color="var(--green-soft)" style={{ margin: '0 auto 12px' }} />
        <div style={{ color: 'var(--text-primary)', fontWeight: 500, marginBottom: 6 }}>
          拖拽文件到此，或点击选择（支持多选）
        </div>
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          支持 PDF、Word（.doc/.docx）、Excel（.xlsx/.xls）
        </div>
        <input
          ref={fileInputRef} type="file" hidden multiple
          accept=".pdf,.doc,.docx,.xlsx,.xls"
          onChange={e => {
            const files = Array.from(e.target.files || [])
            if (files.length > 0) openModal(files)
            e.target.value = ''
          }}
        />
      </div>

      {/* 任务列表 */}
      {tasks.map(task => (
        <div key={task.taskId} className="glass-card" style={{ padding: '14px 18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <FileText size={16} color="var(--green-muted)" />
              <div>
                <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{task.filename}</div>
                <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>→ 知识库：{task.collection}</div>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {task.status === 'done'  && <CheckCircle size={16} color="var(--accent)" />}
              {task.status === 'error' && <AlertCircle size={16} color="var(--danger)" />}
              {(task.status === 'done' || task.status === 'error') && (
                <button className="btn-ghost" style={{ padding: '2px 6px', fontSize: 11 }}
                  onClick={() => removeTask(task.taskId)}>
                  <X size={12} />
                </button>
              )}
            </div>
          </div>

          <div className="progress-bar-bg">
            <div
              className="progress-bar-fill"
              style={{
                width: `${task.percent}%`,
                animation: task.status === 'done' ? 'none' : undefined,
                background: task.status === 'done'
                  ? 'var(--accent)'
                  : task.status === 'error'
                  ? 'var(--danger)'
                  : undefined,
              }}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
            <span style={{ fontSize: 11, color: task.status === 'error' ? 'var(--danger)' : 'var(--text-secondary)' }}>
              {task.status === 'error' ? task.error : task.step}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-dim)', fontVariantNumeric: 'tabular-nums' }}>
              {task.status === 'done' && task.chunks ? `${task.chunks} chunks` : `${task.percent}%`}
            </span>
          </div>
        </div>
      ))}

      {/* 上传弹窗 */}
      {showModal && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(200,230,195,0.5)',
          backdropFilter: 'blur(6px)', display: 'flex',
          alignItems: 'center', justifyContent: 'center', zIndex: 100,
        }}>
          <div className="glass-card" style={{ padding: 28, width: 440, maxWidth: '90vw' }}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 18, color: 'var(--text-primary)' }}>
              上传到知识库
            </div>

            {/* 文件列表预览 */}
            <div style={{ marginBottom: 14, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {pendingFiles.map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <FileText size={12} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                </div>
              ))}
              {pendingFiles.length > 1 && (
                <div style={{ fontSize: 11, color: 'var(--green-muted)', marginTop: 2 }}>
                  共 {pendingFiles.length} 个文件，将全部追加到同一知识库
                </div>
              )}
            </div>

            {/* 知识库名称 */}
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>知识库名称</div>
            <input
              value={collectionName}
              onChange={e => setCollectionName(e.target.value)}
              placeholder="如：hydro_manual_en"
              style={{ marginBottom: 16 }}
              autoFocus
              onKeyDown={e => e.key === 'Enter' && handleUpload()}
            />

            {/* Embedding 模型选择 */}
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>Embedding 模型</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
              {EMBEDDING_OPTIONS.map(opt => (
                <label key={opt.value} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer',
                  padding: '8px 12px', borderRadius: 8,
                  background: embeddingModel === opt.value ? 'rgba(122,158,114,0.15)' : 'rgba(255,255,255,0.3)',
                  border: embeddingModel === opt.value ? '1px solid var(--green-muted)' : '1px solid var(--glass-border)',
                  transition: 'all 0.15s',
                }}>
                  <input
                    type="radio" name="embeddingModel"
                    value={opt.value}
                    checked={embeddingModel === opt.value}
                    onChange={() => setEmbeddingModel(opt.value)}
                    style={{ marginTop: 2, accentColor: 'var(--accent)', cursor: 'pointer' }}
                  />
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{opt.label}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{opt.hint}</div>
                  </div>
                </label>
              ))}
            </div>

            {/* VLM OCR */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox" checked={useVlmOcr}
                onChange={e => setUseVlmOcr(e.target.checked)}
                style={{ width: 'auto', cursor: 'pointer' }}
              />
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                启用 VLM 表格 OCR（识别图片表格，消耗 API）
              </span>
            </label>

            {/* 父子切分 */}
            <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 20, marginTop: 10, cursor: 'pointer' }}>
              <input
                type="checkbox" checked={useParentChild}
                onChange={e => setUseParentChild(e.target.checked)}
                style={{ width: 'auto', cursor: 'pointer', marginTop: 2 }}
              />
              <div>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                  启用父子切分（推荐新库使用）
                </span>
                <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
                  小 chunk 用于精准检索，大 chunk（整节）送入 LLM，改善并列定义、步骤列表等被截断的问题
                </div>
              </div>
            </label>

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button className="btn-ghost" onClick={() => setShowModal(false)}>取消</button>
              <button className="btn-primary" onClick={handleUpload} disabled={!collectionName.trim()}>
                {pendingFiles.length > 1 ? `开始处理（${pendingFiles.length} 个文件）` : '开始处理'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
