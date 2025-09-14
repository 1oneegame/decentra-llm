import { useState, useEffect } from "react"
import { GridItem } from "@/components/grid"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

interface Client {
  client_code: number
  name: string
  status: string
  age: number
  city: string
  avg_monthly_balance_KZT: number
}

interface QueuedMessage {
  id: string
  client: Client
  message: string
  scheduledTime: string
  status: "pending" | "sent" | "failed" | "approved" | "rejected"
  isMLGenerated: boolean
  confidence?: number
  product?: string
}


interface QueueTabProps {
  clients: Client[]
}

export default function QueueTab({ clients }: QueueTabProps) {
  const [queuedMessages, setQueuedMessages] = useState<QueuedMessage[]>([])
  const [isGeneratingQueue, setIsGeneratingQueue] = useState(false)

  const generateMLQueue = async () => {
    setIsGeneratingQueue(true)
    
    try {
      const response = await fetch('http://localhost:8000/predictions-batch?add_randomness=true')
      
      if (response.ok) {
        const batchData = await response.json()
        const predictions = batchData.predictions || []
        
        const mlMessages: QueuedMessage[] = []
        
        for (const prediction of predictions) {
          const client = clients.find(c => c.client_code === prediction.client_code)
          if (client) {
            const now = new Date()
            const scheduledTime = new Date(now.getTime() + Math.random() * 24 * 60 * 60 * 1000)
            scheduledTime.setHours(
              19 + Math.floor(Math.random() * 3), 
              Math.floor(Math.random() * 60), 
              0, 0
            )
            
            mlMessages.push({
              id: `ml-${client.client_code}-${Date.now()}-${Math.random()}`,
              client,
              message: prediction.push_notification,
              scheduledTime: scheduledTime.toISOString(),
              status: "pending",
              isMLGenerated: true,
              confidence: prediction.confidence,
              product: prediction.recommended_product
            })
          }
        }
        
        setQueuedMessages(mlMessages)
        console.log(`🔄 Регенерирована очередь через batch API: ${mlMessages.length} новых сообщений`)
        
      } else {
        throw new Error('Batch API недоступен')
      }
      
    } catch (error) {
      console.error('Ошибка генерации очереди:', error)
      
      const fallbackMessages: QueuedMessage[] = clients.map(client => {
        const fallbackTime = new Date()
        fallbackTime.setHours(20, Math.floor(Math.random() * 60), 0, 0)
        
        return {
          id: `fallback-${client.client_code}-${Date.now()}`,
          client,
          message: `${client.name}, у нас есть персональное предложение для вас! Узнайте подробнее.`,
          scheduledTime: fallbackTime.toISOString(),
          status: "pending",
          isMLGenerated: true,
          confidence: 0.5,
          product: "Персональное предложение"
        }
      })
      
      setQueuedMessages(fallbackMessages)
      console.log(`🔄 Использован fallback: ${fallbackMessages.length} сообщений`)
      
    } finally {
      setIsGeneratingQueue(false)
    }
  }

  useEffect(() => {
    if (clients.length > 0 && queuedMessages.length === 0) {
      generateMLQueue()
    }
  }, [clients.length])

  const removeFromQueue = (messageId: string) => {
    setQueuedMessages(prev => prev.filter(msg => msg.id !== messageId))
  }

  const approveMessage = (messageId: string) => {
    setQueuedMessages(prev => 
      prev.map(msg => 
        msg.id === messageId ? { ...msg, status: "approved" as const } : msg
      )
    )
  }

  const rejectMessage = (messageId: string) => {
    setQueuedMessages(prev => 
      prev.map(msg => 
        msg.id === messageId ? { ...msg, status: "rejected" as const } : msg
      )
    )
  }

  const editMessage = (messageId: string, newMessage: string) => {
    setQueuedMessages(prev => 
      prev.map(msg => 
        msg.id === messageId ? { ...msg, message: newMessage, isMLGenerated: false } : msg
      )
    )
  }

  return (
    <>
      <GridItem className="col-span-3">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-xl font-semibold text-gray-900">AI Message Queue</h3>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500 font-mono">
              {queuedMessages.length} messages • {queuedMessages.filter(m => m.status === 'approved').length} approved
            </span>
            <Button 
              onClick={generateMLQueue}
              disabled={isGeneratingQueue || clients.length === 0}
              size="sm"
              className="bg-blue-900 hover:bg-blue-800 text-white"
            >
              {isGeneratingQueue ? "Generating..." : "Regenerate Queue"}
            </Button>
          </div>
        </div>
      
      {queuedMessages.length === 0 ? (
        <div className="text-center py-16 text-gray-500 border-t border-gray-100">
          <div className="mb-4">🤖</div>
          <p className="text-sm">AI Message Queue is empty</p>
          <p className="text-xs text-gray-400 mt-2">Queue will auto-populate when ML recommendations are available</p>
          {isGeneratingQueue && (
            <div className="inline-flex items-center text-blue-600 mt-4">
              <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin mr-2"></div>
              Generating queue...
            </div>
          )}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-gray-100">
                <TableHead className="text-gray-600">Client</TableHead>
                <TableHead className="text-gray-600">Product</TableHead>
                <TableHead className="text-gray-600">Message</TableHead>
                <TableHead className="text-gray-600">Confidence</TableHead>
                <TableHead className="text-gray-600">Scheduled</TableHead>
                <TableHead className="text-gray-600">Status</TableHead>
                <TableHead className="text-gray-600">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {queuedMessages.map((msg) => (
                <TableRow key={msg.id} className="border-gray-100 hover:bg-gray-50">
                  <TableCell>
                    <div className="font-medium text-gray-900">{msg.client.name}</div>
                    <div className="text-sm text-gray-500 font-mono">#{msg.client.client_code}</div>
                    <div className="mt-1">
                      <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-50 text-blue-900">
                        🤖 AI
                      </span>
                    </div>
                  </TableCell>
                  
                  <TableCell>
                    <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-50 text-gray-700">
                      {msg.product}
                    </span>
                  </TableCell>
                  
                  <TableCell>
                    <textarea
                      value={msg.message}
                      onChange={(e) => editMessage(msg.id, e.target.value)}
                      className="w-full p-2 text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      disabled={msg.status === 'sent'}
                      rows={2}
                    />
                  </TableCell>
                  
                  <TableCell>
                    {msg.confidence && (
                      <div className="text-center">
                        <div className="text-sm font-medium text-gray-900 mb-1">
                          {(msg.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-900 h-1.5 rounded-full transition-all"
                            style={{ width: `${msg.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </TableCell>
                  
                  <TableCell className="font-mono text-sm text-gray-700">
                    <div>{new Date(msg.scheduledTime).toLocaleDateString()}</div>
                    <div className="text-xs text-gray-500">{new Date(msg.scheduledTime).toLocaleTimeString()}</div>
                  </TableCell>
                  
                  <TableCell>
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                      msg.status === 'pending' 
                        ? 'bg-yellow-50 text-yellow-700' 
                        : msg.status === 'approved'
                        ? 'bg-green-50 text-green-700'
                        : msg.status === 'rejected'
                        ? 'bg-red-50 text-red-700'
                        : msg.status === 'sent'
                        ? 'bg-blue-50 text-blue-700'
                        : 'bg-gray-50 text-gray-700'
                    }`}>
                      {msg.status.charAt(0).toUpperCase() + msg.status.slice(1)}
                    </span>
                  </TableCell>
                  
                  <TableCell>
                    <div className="flex items-center gap-1">
                      {msg.status === 'pending' && (
                        <>
                          <Button
                            size="sm"
                            onClick={() => approveMessage(msg.id)}
                            className="bg-blue-900 hover:bg-blue-800 text-white"
                          >
                            ✓
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => rejectMessage(msg.id)}
                            className="border-gray-200 text-gray-600 hover:text-gray-900"
                          >
                            ✗
                          </Button>
                        </>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => removeFromQueue(msg.id)}
                        className="border-gray-200 text-gray-600 hover:text-gray-900"
                      >
                        🗑
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
    </>
  )
}
