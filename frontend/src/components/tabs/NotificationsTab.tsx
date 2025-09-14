import { GridItem } from "@/components/grid"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useState, useEffect } from "react"

interface Recommendation {
  client_code: number
  product: string
  confidence: number
  expected_benefit: number
  cluster_description: string
  push_notification: string
}

interface GeneratedMessage {
  id: string
  client_code: number
  client_name: string
  message: string
  product: string
  confidence: number
  expected_benefit: number
  optimal_time: number
  timestamp: string
  is_regenerated?: boolean
}

interface GenerationStats {
  total_generated: number
  avg_confidence: number
  unique_products: number
  generated_today: number
}

interface NotificationsTabProps {
  recommendations: Recommendation[]
}

export default function NotificationsTab({ recommendations }: NotificationsTabProps) {
  const [generatedMessages, setGeneratedMessages] = useState<GeneratedMessage[]>([])
  const [stats, setStats] = useState<GenerationStats>({
    total_generated: 0,
    avg_confidence: 0,
    unique_products: 0,
    generated_today: 0
  })
  const [isGenerating, setIsGenerating] = useState<string | null>(null)
  const [clients, setClients] = useState<any[]>([])
  const [selectedClients, setSelectedClients] = useState<number[]>([])

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const response = await fetch('http://localhost:8000/clients')
        if (response.ok) {
          const data = await response.json()
          setClients(data)
        }
      } catch (error) {
        console.error('Error loading clients:', error)
      }
    }
    fetchClients()
  }, [])

  useEffect(() => {
    const calculateStats = () => {
      const totalGenerated = generatedMessages.length
      const avgConfidence = totalGenerated > 0 
        ? generatedMessages.reduce((sum, m) => sum + m.confidence, 0) / totalGenerated * 100 
        : 0
      
      const uniqueProducts = new Set(generatedMessages.map(m => m.product)).size
      
      const today = new Date().toDateString()
      const generatedToday = generatedMessages.filter(m => 
        new Date(m.timestamp).toDateString() === today
      ).length

      setStats({
        total_generated: totalGenerated,
        avg_confidence: avgConfidence,
        unique_products: uniqueProducts,
        generated_today: generatedToday
      })
    }
    calculateStats()
  }, [generatedMessages])

  const generateMessage = async (clientCode: number) => {
    setIsGenerating(clientCode.toString())
    
    try {
      const response = await fetch(`http://localhost:8000/predict-push/${clientCode}?add_randomness=true`, {
        method: "POST"
      })
      
      if (response.ok) {
        const data = await response.json()
        const client = clients.find(c => c.client_code === clientCode)
        
        const newMessage: GeneratedMessage = {
          id: `msg-${Date.now()}-${clientCode}`,
          client_code: clientCode,
          client_name: client?.name || `Client ${clientCode}`,
          message: data.push_notification,
          product: data.recommended_product,
          confidence: data.confidence,
          expected_benefit: data.expected_benefit,
          optimal_time: data.optimal_time,
          timestamp: new Date().toISOString(),
          is_regenerated: false
        }
        
        setGeneratedMessages(prev => [newMessage, ...prev])
      }
    } catch (error) {
      console.error("Failed to generate message:", error)
    } finally {
      setIsGenerating(null)
    }
  }

  const regenerateMessage = async (messageId: string) => {
    const existingMessage = generatedMessages.find(m => m.id === messageId)
    if (!existingMessage) return
    
    setIsGenerating(messageId)
    
    try {
      const response = await fetch(`http://localhost:8000/predict-push/${existingMessage.client_code}?add_randomness=true`, {
        method: "POST"
      })
      
      if (response.ok) {
        const data = await response.json()
        
        setGeneratedMessages(prev => 
          prev.map(m => 
            m.id === messageId 
              ? {
                  ...m,
                  message: data.push_notification,
                  product: data.recommended_product,
                  confidence: data.confidence,
                  expected_benefit: data.expected_benefit,
                  optimal_time: data.optimal_time,
                  timestamp: new Date().toISOString(),
                  is_regenerated: true
                }
              : m
          )
        )
      }
    } catch (error) {
      console.error("Failed to regenerate message:", error)
    } finally {
      setIsGenerating(null)
    }
  }

  const generateBatchMessages = async () => {
    if (selectedClients.length === 0) return
    
    setIsGenerating('batch')
    
    for (const clientCode of selectedClients) {
      await generateMessage(clientCode)
      await new Promise(resolve => setTimeout(resolve, 300))
    }
    
    setIsGenerating(null)
    setSelectedClients([])
  }

  const toggleClientSelection = (clientCode: number) => {
    setSelectedClients(prev => 
      prev.includes(clientCode) 
        ? prev.filter(c => c !== clientCode)
        : [...prev, clientCode]
    )
  }

  return (
    <>
      <GridItem className="col-span-2">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-xl font-semibold text-gray-900">Сгенерированные сообщения</h3>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500 font-mono">
              {generatedMessages.length} сгенерировано • {selectedClients.length} выбранных
            </span>
            <Button 
              onClick={generateBatchMessages}
              disabled={isGenerating === 'batch' || selectedClients.length === 0}
              size="sm"
              className="bg-blue-900 hover:bg-blue-800 text-white"
            >
              {isGenerating === 'batch' ? "Генерация..." : `Сгенерировать для выбранных (${selectedClients.length})`}
            </Button>
          </div>
        </div>

        {generatedMessages.length === 0 ? (
          <div className="text-center py-16 text-gray-500 border-t border-gray-100">
            <div className="mb-4">✍️</div>
            <p className="text-lg font-medium mb-2">Нет сообщений сгенерировано</p>
            <p className="text-sm mb-6">Сгенерируйте персонализированные сообщения для конкретных клиентов</p>
            <div className="space-y-4">
              <h4 className="font-medium text-gray-700">Выберите клиентов для генерации сообщений:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
                {clients.slice(0, 9).map((client) => (
                  <div 
                    key={client.client_code} 
                    className={`p-4 rounded border cursor-pointer transition-all ${
                      selectedClients.includes(client.client_code)
                        ? 'bg-blue-50 border-blue-200 ring-2 ring-blue-500'
                        : 'bg-gray-50 border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => toggleClientSelection(client.client_code)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">{client.name}</span>
                      <input
                        type="checkbox"
                        checked={selectedClients.includes(client.client_code)}
                        onChange={() => {}} 
                        className="w-4 h-4 text-blue-600"
                      />
                    </div>
                    <div className="text-sm text-gray-600">
                      <div>#{client.client_code} • {client.city}</div>
                      <div>₸ {client.avg_monthly_balance_KZT?.toLocaleString()}</div>
                    </div>
                    <Button
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        generateMessage(client.client_code)
                      }}
                      disabled={isGenerating === client.client_code.toString()}
                      className="w-full mt-3 bg-blue-900 hover:bg-blue-800 text-white"
                    >
                      {isGenerating === client.client_code.toString() ? "Генерация..." : "Сгенерировать сообщение"}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-gray-100">
                  <TableHead className="text-gray-600">Клиент</TableHead>
                  <TableHead className="text-gray-600">Сообщение</TableHead>
                  <TableHead className="text-gray-600">Продукт</TableHead>
                  <TableHead className="text-gray-600">Уверенность</TableHead>
                  <TableHead className="text-gray-600">Ожидаемая выгода</TableHead>
                  <TableHead className="text-gray-600">Действия</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {generatedMessages.slice(0, 10).map((message) => (
                  <TableRow key={message.id} className="border-gray-100 hover:bg-gray-50">
                    <TableCell>
                      <div className="font-medium text-gray-900">{message.client_name}</div>
                      <div className="text-sm text-gray-500 font-mono">#{message.client_code}</div>
                      {message.is_regenerated && (
                        <div className="mt-1">
                          <span className="inline-flex items-center px-1 py-0.5 rounded text-xs font-medium bg-orange-50 text-orange-700">
                            🔄 Перегенерировано
                          </span>
                        </div>
                      )}
                    </TableCell>
                    
                    <TableCell className="max-w-xs">
                      <p className="text-sm text-gray-700 line-clamp-3">"{message.message}"</p>
                      <div className="text-xs text-gray-500 mt-1 font-mono">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-50 text-gray-700">
                        {message.product}
                      </span>
                    </TableCell>
                    
                    <TableCell>
                      <div className="text-center">
                        <div className="text-sm font-medium text-gray-900 mb-1">
                          {(message.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-900 h-1.5 rounded-full"
                            style={{ width: `${message.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      <div className="text-center">
                        <div className="text-sm font-medium text-gray-900">
                          ₸ {message.expected_benefit.toLocaleString()}
                        </div>
                        <div className="text-xs text-gray-500">
                          Оптимальное время: {message.optimal_time}:00
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => regenerateMessage(message.id)}
                          disabled={isGenerating === message.id}
                          className="border-gray-200 text-gray-600 hover:text-gray-900"
                        >
                          {isGenerating === message.id ? "..." : "🔄"}
                        </Button>
                        <Button
                          size="sm"
                          className="bg-blue-900 hover:bg-blue-800 text-white"
                          onClick={() => {
                            navigator.clipboard.writeText(message.message)
                            const btn = document.activeElement as HTMLButtonElement
                            const originalText = btn.textContent
                            btn.textContent = '✓'
                            setTimeout(() => {
                              btn.textContent = originalText
                            }, 1000)
                          }}
                        >
                          Копировать
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </GridItem>
      
      <GridItem>
        <h3 className="text-xl font-semibold mb-8 text-gray-900">Статистика генерации</h3>
        <div className="space-y-8">
          <div className="text-center border-b border-gray-100 pb-6">
            <div className="text-4xl font-bold mb-2 text-blue-900">{stats.generated_today}</div>
            <div className="text-sm text-gray-600">Сгенерировано сегодня</div>
          </div>
          <div className="text-center border-b border-gray-100 pb-6">
            <div className="text-4xl font-bold mb-2 text-blue-900">{stats.unique_products}</div>
            <div className="text-sm text-gray-600">Уникальные продукты</div>
          </div>
          <div className="text-center border-b border-gray-100 pb-6">
            <div className="text-4xl font-bold mb-2 text-blue-900">{stats.avg_confidence.toFixed(1)}%</div>
            <div className="text-sm text-gray-600">Средняя уверенность</div>
          </div>
          <div className="text-center pt-2">
            <div className="text-2xl font-bold mb-2 text-gray-700">{stats.total_generated}</div>
            <div className="text-sm text-gray-600">Всего сгенерировано</div>
          </div>
        </div>
      </GridItem>
    </>
  )
}
