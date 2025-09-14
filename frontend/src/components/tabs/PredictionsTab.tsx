import { GridItem } from "@/components/grid"
import { useState, useEffect } from "react"

interface MLMetrics {
  model_status: string
  classifier_accuracy: number
  regressor_rmse: number
  clustering_score: number
  clusters_count: number
  dataset_shape: [number, number]
  feature_count: number
}

interface BusinessAnalytics {
  top_product: {
    name: string
    percentage: number
  }
  average_expected_benefit: number
  optimal_time: {
    hour: number
    minute: number
    formatted: string
  }
}

interface PredictionsTabProps {
  mlMetrics: MLMetrics | null
}

export default function PredictionsTab({ mlMetrics }: PredictionsTabProps) {
  const [businessAnalytics, setBusinessAnalytics] = useState<BusinessAnalytics | null>(null)
  const [isLoadingAnalytics, setIsLoadingAnalytics] = useState(false)

  useEffect(() => {
    const fetchBusinessAnalytics = async () => {
      setIsLoadingAnalytics(true)
      try {
        const response = await fetch('http://localhost:8000/business-analytics')
        if (response.ok) {
          const data = await response.json()
          setBusinessAnalytics(data)
        }
      } catch (error) {
        console.error('Ошибка загрузки бизнес-аналитики:', error)
      } finally {
        setIsLoadingAnalytics(false)
      }
    }

    fetchBusinessAnalytics()
  }, [])

  return (
    <>
      <GridItem>
        <h3 className="text-xl font-semibold mb-8 text-gray-900">Производительность модели</h3>
        {mlMetrics ? (
          <div className="space-y-8">
            <div className="grid grid-cols-2 gap-8">
              <div className="text-center py-8 border-t border-gray-100">
                <div className="text-5xl font-bold mb-2 text-blue-900">
                  {mlMetrics.classifier_accuracy ? (mlMetrics.classifier_accuracy * 100).toFixed(1) : '0.0'}%
                </div>
                <div className="text-sm text-gray-600">Точность</div>
              </div>
              <div className="text-center py-8 border-t border-gray-100">
                <div className="text-5xl font-bold mb-2 text-blue-900">
                  {mlMetrics.clusters_count || 0}
                </div>
                <div className="text-sm text-gray-600">Кластеры</div>
              </div>
            </div>
            <div className="space-y-6 border-t border-gray-100 pt-8">
              <div className="flex justify-between items-center py-3 border-b border-gray-100">
                <span className="text-sm text-gray-600">RMSE</span>
                <span className="font-mono text-xl text-gray-900">
                  {mlMetrics.regressor_rmse ? mlMetrics.regressor_rmse.toFixed(0) : '0'}
                </span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-gray-100">
                <span className="text-sm text-gray-600">Оценка кластеризации</span>
                <span className="font-mono text-xl text-gray-900">
                  {mlMetrics.clustering_score ? (mlMetrics.clustering_score * 100).toFixed(1) : '0.0'}%
                </span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-gray-100">
                <span className="text-sm text-gray-600">Признаки</span>
                <span className="font-mono text-xl text-gray-900">{mlMetrics.feature_count || 0}</span>
              </div>
              <div className="flex justify-between items-center py-3">
                <span className="text-sm text-gray-600">Размер датасета</span>
                <span className="font-mono text-xl text-gray-900">
                  {mlMetrics.dataset_shape ? mlMetrics.dataset_shape[0] : 0} клиентов
                </span>
              </div>
            </div>
            {mlMetrics.model_status === 'not_trained' && (
              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  ⚠️ Модель обучается автоматически при первом запуске сервера
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <p className="text-sm">Загрузка метрик модели...</p>
          </div>
        )}
      </GridItem>
      
      <GridItem>
        <h3 className="text-xl font-semibold mb-8 text-gray-900">Бизнес-аналитика</h3>
        {isLoadingAnalytics ? (
          <div className="text-center py-12 text-gray-500">
            <div className="w-4 h-4 border-2 border-neutral-300 border-t-neutral-900 rounded-full animate-spin mx-auto mb-3"></div>
            <p className="text-sm">Загрузка аналитики...</p>
          </div>
        ) : businessAnalytics ? (
          <div className="space-y-8">
            <div className="border-t border-gray-100 pt-6">
              <div className="text-sm font-medium text-gray-600 mb-3">Топ продукт</div>
              <div className="text-lg font-semibold text-gray-900">{businessAnalytics.top_product.name}</div>
              <div className="text-sm text-gray-500 mt-1">{businessAnalytics.top_product.percentage}% от всех предсказаний</div>
            </div>
            <div className="border-t border-gray-100 pt-6">
              <div className="text-sm font-medium text-gray-600 mb-3">Средняя ожидаемая выгода</div>
              <div className="text-2xl font-mono font-bold text-blue-900">
                ₸ {businessAnalytics.average_expected_benefit.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500 mt-1">на клиента</div>
            </div>
            <div className="border-t border-gray-100 pt-6">
              <div className="text-sm font-medium text-gray-600 mb-3">Оптимальное время</div>
              <div className="text-lg font-semibold text-gray-900">{businessAnalytics.optimal_time.formatted}</div>
              <div className="text-sm text-gray-500 mt-1">наивысшая активность</div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <p className="text-sm">Ошибка загрузки аналитики</p>
          </div>
        )}
      </GridItem>
    </>
  )
}
