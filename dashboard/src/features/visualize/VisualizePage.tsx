import { useTranslation } from "react-i18next"
import { ProGate } from "@/components/common/ProGate"
import MemoryChart from "@/components/common/MemoryChart"

export default function VisualizePage() {
  const { t } = useTranslation()

  return (
    <ProGate label={t("license.pro_feature", "Pro Feature")}>
      <div className="space-y-6 p-6">
        <div>
          <h1 className="font-display text-2xl font-bold">
            {t("visualize.title", "Memory Visualizer")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("visualize.description", "Query your memories and generate charts from stored data.")}
          </p>
        </div>

        <MemoryChart />
      </div>
    </ProGate>
  )
}
