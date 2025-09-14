"use client";

import { useState, useEffect } from "react";
import { Grid, GridItem } from "@/components/grid";
import { Button } from "@/components/ui/button";
import ClientsTab from "@/components/tabs/ClientsTab";
import NotificationsTab from "@/components/tabs/NotificationsTab";
import QueueTab from "@/components/tabs/QueueTab";
import PredictionsTab from "@/components/tabs/PredictionsTab";

interface Client {
  client_code: number;
  name: string;
  status: string;
  age: number;
  city: string;
  avg_monthly_balance_KZT: number;
}

interface Recommendation {
  client_code: number;
  product: string;
  confidence: number;
  expected_benefit: number;
  cluster_description: string;
  push_notification: string;
}

interface MLMetrics {
  model_status: string;
  classifier_accuracy: number;
  regressor_rmse: number;
  clustering_score: number;
  clusters_count: number;
  dataset_shape: [number, number];
  feature_count: number;
}

type TabId = "clients" | "notifications" | "queue" | "predictions";

export default function Home() {
  const [clients, setClients] = useState<Client[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [mlMetrics, setMlMetrics] = useState<MLMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>("clients");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [clientsRes, recommendationsRes, metricsRes] = await Promise.all([
        fetch("http://localhost:8000/clients"),
        fetch("http://localhost:8000/recommendations"),
        fetch("http://localhost:8000/ml-metrics")
      ]);

      if (clientsRes.ok) {
        const clientsData = await clientsRes.json();
        setClients(clientsData);
      }

      if (recommendationsRes.ok) {
        const recommendationsData = await recommendationsRes.json();
        setRecommendations(recommendationsData);
      }

      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        console.log('ML Metrics data:', metricsData);
        setMlMetrics(metricsData.ml_metrics);
      } else {
        console.error('Failed to fetch ML metrics:', metricsRes.status);
      }
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  };


  const tabs = [
    { id: "clients" as TabId, label: "Клиенты" },
    { id: "notifications" as TabId, label: "Уведомления" },
    { id: "queue" as TabId, label: "Очередь" },
    { id: "predictions" as TabId, label: "Аналитика" }

  ];

  const renderTabContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center min-h-48ing bg-white">
          <div className="text-center">
            <div className="w-4 h-4 border-2 border-neutral-300 border-t-neutral-900 rounded-full animate-spin mx-auto mb-3"></div>
            <p className="text-neutral-500 text-sm">Загружаем...</p>
          </div>
        </div>
      );
    }

    switch (activeTab) {
      case "clients":
        return <ClientsTab clients={clients} />;
      case "notifications":
        return <NotificationsTab recommendations={recommendations} />;
      case "queue":
        return <QueueTab clients={clients} />;
      case "predictions":
        return <PredictionsTab mlMetrics={mlMetrics} />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-white text-black my-14">
      <section className="container mx-auto px-4 pt-10">
        <Grid>
          <GridItem className="flex flex-col items-center justify-center py-16 md:py-24">
            <div className="max-w-3xl mx-auto text-center">
              <h1 className="text-4xl md:text-6xl font-bold mb-6 text-gray-900">Decentra</h1>
              <p className="text-xl md:text-2xl text-gray-600 mb-10">
                Система персонализированных пуш-уведомлений с использованием ML
              </p>
              <div className="flex items-center justify-center gap-8 mb-10">
                <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
                  mlMetrics?.model_status === 'trained' 
                    ? 'bg-green-50 text-green-700 border border-green-200' 
                    : 'bg-yellow-50 text-yellow-700 border border-yellow-200'
                }`}>
                  Статус ML: {mlMetrics?.model_status === 'trained' ? 'Готов' : 'Загрузка'}
                </div>
                <div className="text-sm text-gray-600 font-mono">
                  {clients.length} клиентов загружено
                </div>
              </div>
            </div>
          </GridItem>
        </Grid>
      </section>

      <section className="container mx-auto px-4 py-0">
        <Grid connectTo="bottom" noDecoration="top">
          <GridItem className="text-center py-16">
            <h2 className="text-3xl font-bold mb-4 text-gray-900">Личный кабинет</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">Управление аналитикой клиентов и пуш-уведомлениями с использованием ML</p>
          </GridItem>
        </Grid>
        <Grid connectTo="top">
          <GridItem>
            <div className="border-b border-gray-100">
              <nav className="flex justify-center space-x-8">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-4 px-2 text-sm font-medium border-b-2 transition-colors duration-200 ${
                      activeTab === tab.id
                        ? 'border-blue-900 text-blue-900'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </GridItem>
        </Grid>
      </section>

      <section className="container mx-auto px-4 pb-10">
        {activeTab === "clients" ? (
          <Grid connectTo="top">
            {renderTabContent()}
          </Grid>
        ) : activeTab === "queue" ? (
          <Grid columns={3} connectTo="top">
            {renderTabContent()}
          </Grid>
        ) : (
          <Grid columns={2} connectTo="top">
            {renderTabContent()}
          </Grid>
        )}
      </section>
    </div>
  );
}
