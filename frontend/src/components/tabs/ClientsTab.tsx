import { useState } from "react"
import { GridItem } from "@/components/grid"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Modal } from "@/components/ui/modal"

interface Client {
  client_code: number
  name: string
  status: string
  age: number
  city: string
  avg_monthly_balance_KZT: number
}

interface ClientAnalysis {
  client: Client
  recommendation: any
  pushPrediction: any
}

interface ClientsTabProps {
  clients: Client[]
}

export default function ClientsTab({ clients }: ClientsTabProps) {
  const [selectedClient, setSelectedClient] = useState<Client | null>(null)
  const [clientAnalysis, setClientAnalysis] = useState<ClientAnalysis | null>(null)
  const [analyzingClient, setAnalyzingClient] = useState(false)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const analyzeClient = async (client: Client) => {
    setSelectedClient(client)
    setAnalyzingClient(true)
    setClientAnalysis(null)
    setIsModalOpen(true)
    
    try {
      const [recommendationRes, pushRes] = await Promise.all([
        fetch(`http://localhost:8000/recommendations/${client.client_code}`),
        fetch(`http://localhost:8000/predict-push/${client.client_code}`, { method: "POST" })
      ])

      const analysis: ClientAnalysis = {
        client: client,
        recommendation: null,
        pushPrediction: null
      }

      if (recommendationRes.ok) {
        analysis.recommendation = await recommendationRes.json()
      }

      if (pushRes.ok) {
        analysis.pushPrediction = await pushRes.json()
      }

      setClientAnalysis(analysis)
    } catch (error) {
      console.error("Failed to analyze client:", error)
      alert("Error analyzing client")
    } finally {
      setAnalyzingClient(false)
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('kk-KZ', {
      style: 'currency',
      currency: 'KZT',
      minimumFractionDigits: 0
    }).format(amount)
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "Премиальный клиент":
        return "default"
      case "Зарплатный клиент":
        return "secondary"
      case "Студент":
        return "outline"
      default:
        return "outline"
    }
  }

  const closeModal = () => {
    setIsModalOpen(false)
    setSelectedClient(null)
    setClientAnalysis(null)
  }

  return (
    <>
      <GridItem>
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-xl font-semibold text-gray-900">Clients Database</h3>
          <span className="text-sm text-gray-500 font-mono">{clients.length} total clients</span>
        </div>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-gray-100">
                <TableHead className="text-gray-600">Name</TableHead>
                <TableHead className="text-gray-600">Status</TableHead>
                <TableHead className="text-gray-600">Age</TableHead>
                <TableHead className="text-gray-600">City</TableHead>
                <TableHead className="text-gray-600">Balance</TableHead>
                <TableHead className="text-gray-600"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {clients.map((client) => (
                <TableRow key={client.client_code} className="border-gray-100 hover:bg-gray-50">
                  <TableCell className="font-medium text-gray-900">{client.name}</TableCell>
                  <TableCell>
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                      client.status === "Премиальный клиент" 
                        ? "bg-blue-50 text-blue-900" 
                        : client.status === "Зарплатный клиент"
                        ? "bg-green-50 text-green-900"
                        : "bg-gray-50 text-gray-700"
                    }`}>
                      {client.status}
                    </span>
                  </TableCell>
                  <TableCell className="text-gray-700">{client.age}</TableCell>
                  <TableCell className="text-gray-700">{client.city}</TableCell>
                  <TableCell className="font-mono text-sm text-gray-900">
                    {formatCurrency(client.avg_monthly_balance_KZT)}
                  </TableCell>
                  <TableCell>
                    <Button 
                      size="sm"
                      variant="outline"
                      onClick={() => analyzeClient(client)}
                      disabled={analyzingClient}
                      className="border-gray-200 text-gray-600 hover:text-gray-900"
                    >
                     Просмотр
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </GridItem>

      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title={selectedClient ? `Analysis: ${selectedClient.name}` : "Client Analysis"}
      >
        {analyzingClient ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-900 rounded-full animate-spin mx-auto mb-3"></div>
              <p className="text-sm text-gray-500">Анализируем клиента...</p>
            </div>
          </div>
        ) : selectedClient && clientAnalysis ? (
          <div className="space-y-8">
            <div className="border-b border-gray-100 pb-6">
              <h4 className="text-xl font-semibold text-gray-900 mb-4">{selectedClient.name}</h4>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-sm text-gray-600 mb-2">Статус</div>
                  <span className={`inline-flex items-center px-3 py-1 rounded text-sm font-medium ${
                    selectedClient.status === "Премиальный клиент" 
                      ? "bg-blue-50 text-blue-900" 
                      : selectedClient.status === "Зарплатный клиент"
                      ? "bg-green-50 text-green-900"
                      : "bg-gray-50 text-gray-700"
                  }`}>
                    {selectedClient.status}
                  </span>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-2">Баланс</div>
                  <div className="font-mono text-lg text-gray-900">
                    {formatCurrency(selectedClient.avg_monthly_balance_KZT)}
                  </div>
                </div>
              </div>
            </div>
            
            {clientAnalysis.recommendation && (
              <div className="border-b border-gray-100 pb-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Рекомендация продукта</h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Продукт</span>
                    <span className="font-medium text-gray-900">{clientAnalysis.recommendation.product}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Уверенность</span>
                    <span className="px-3 py-1 bg-gray-100 rounded text-sm font-mono text-gray-900">
                      {(clientAnalysis.recommendation.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Ожидаемая выгода</span>
                    <span className="font-mono text-lg text-gray-900">
                      {formatCurrency(clientAnalysis.recommendation.expected_benefit)}
                    </span>
                  </div>
                  <div className="mt-4">
                    <div className="text-gray-600 mb-2">Анализ кластера</div>
                    <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded">{clientAnalysis.recommendation.cluster_description}</div>
                  </div>
                </div>
              </div>
            )}

            {clientAnalysis.pushPrediction && (
              <div>
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Предпросмотр пуш-уведомления</h4>
                <div className="text-sm bg-blue-50 p-4 rounded border border-blue-200 italic text-blue-900 mb-4">
                  "{clientAnalysis.pushPrediction.push_notification}"
                </div>
                <div className="text-sm text-gray-500 font-mono">
                  Optimal delivery time: {clientAnalysis.pushPrediction.optimal_time}:00
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-400">
            <p className="text-sm">No analysis data available</p>
          </div>
        )}
      </Modal>
    </>
  )
}
