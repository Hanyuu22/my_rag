export default function AboutPage() {
  return (
    <div style={{
      height: '100%', display: 'flex', alignItems: 'center',
      justifyContent: 'center', gap: 40, padding: '20px 10px',
    }}>

      {/* 左侧：羽入大图 */}
      <div style={{
        flexShrink: 0,
        height: 'min(88vh, 600px)',
        display: 'flex', alignItems: 'flex-end',
        filter: 'drop-shadow(0 8px 32px rgba(50,120,40,0.18))',
      }}>
        <img
          src="/avatar_bot2.png"
          alt="Hanyuu"
          style={{
            height: '100%',
            width: 'auto',
            objectFit: 'contain',
            borderRadius: 24,
          }}
        />
      </div>

      {/* 右侧：漫画对话框 */}
      <div style={{ flex: 1, maxWidth: 520, position: 'relative' }}>
        {/* 气泡主体 */}
        <div style={{
          background: 'rgba(255,255,255,0.93)',
          border: '2.5px solid var(--text-primary)',
          borderRadius: 20,
          padding: '28px 32px',
          position: 'relative',
          boxShadow: '4px 4px 0px rgba(30,60,25,0.15)',
          backdropFilter: 'blur(12px)',
        }}>
          {/* 漫画气泡尾巴（朝左指向羽入） */}
          <div style={{
            position: 'absolute',
            left: -22, top: 60,
            width: 0, height: 0,
            borderTop: '12px solid transparent',
            borderBottom: '12px solid transparent',
            borderRight: '22px solid var(--text-primary)',
          }} />
          <div style={{
            position: 'absolute',
            left: -17, top: 62,
            width: 0, height: 0,
            borderTop: '10px solid transparent',
            borderBottom: '10px solid transparent',
            borderRight: '20px solid rgba(255,255,255,0.93)',
          }} />

          {/* 标题 */}
          <div style={{
            fontSize: 18, fontWeight: 800, color: 'var(--green-deep)',
            marginBottom: 18, letterSpacing: '0.02em',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 22 }}>🌸</span>
            你好！我是羽入知识库助手
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, fontSize: 13.5, lineHeight: 1.8, color: 'var(--text-primary)' }}>

            {/* 系统介绍 */}
            <BubbleSection icon="📖" title="我是什么？">
              我是基于 <Tag>RAG（检索增强生成）</Tag> 技术构建的知识库问答系统。
              你可以上传专业文档，我会帮你从中精准检索信息并生成答案——
              就像一个读过所有资料、随时待命的专家助手。
            </BubbleSection>

            {/* 当前功能 */}
            <BubbleSection icon="✅" title="目前可以做什么？">
              <ul style={{ paddingLeft: 18, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <li>上传 PDF / Word / Excel，自动解析入库</li>
                <li>多知识库管理，独立切换互不干扰</li>
                <li>Hybrid 检索（语义 + 关键词）+ BGE 精排</li>
                <li>LangGraph 多步推理，自动改写查询</li>
                <li>流式回答 + Markdown / LaTeX 渲染</li>
                <li>对话历史按知识库保存，刷新不丢失</li>
              </ul>
            </BubbleSection>

            {/* 下一步目标 */}
            <BubbleSection icon="🎯" title="接下来要做什么？">
              <ul style={{ paddingLeft: 18, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <li>Golden Set 系统测试，RAGAS 批量评分</li>
                <li>查无资料时自动走 LLM / 网络搜索兜底</li>
              </ul>
            </BubbleSection>

            {/* 技术栈 */}
            <div style={{ borderTop: '1px solid rgba(100,180,80,0.2)', paddingTop: 12, fontSize: 12, color: 'var(--text-dim)' }}>
              🛠 MinerU · LangChain · LangGraph · BGE · Chroma · FastAPI · React
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function BubbleSection({ icon, title, children }: {
  icon: string; title: string; children: React.ReactNode
}) {
  return (
    <div>
      <div style={{ fontWeight: 700, color: 'var(--green-mid)', marginBottom: 4, fontSize: 13 }}>
        {icon} {title}
      </div>
      <div style={{ paddingLeft: 4 }}>{children}</div>
    </div>
  )
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      background: 'rgba(90,170,76,0.12)', color: 'var(--green-mid)',
      border: '1px solid rgba(90,170,76,0.28)',
      borderRadius: 4, padding: '0 5px', fontSize: 12.5, fontWeight: 600,
    }}>{children}</span>
  )
}
