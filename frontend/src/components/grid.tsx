import type React from "react"
import { cn } from "@/lib/utils"

interface GridProps {
  children: React.ReactNode
  className?: string
  columns?: 1 | 2 | 3 | 4
  fullWidth?: boolean
  minimal?: boolean
  connectTo?: "top" | "bottom" | "both" | "none"
  id?: string
  noDecoration?: "top" | "bottom" | "both" | "none" 
}

export function Grid({
  children,
  className,
  columns = 1,
  fullWidth = false,
  minimal = true,
  connectTo = "none",
  id,
  noDecoration = "none",
}: GridProps) {
  return (
    <div className="relative">
      {fullWidth && (
        <>
          <div 
            className="absolute top-0 h-px bg-neutral-300 z-0"
            style={{ 
              left: 'calc(-50vw + 50%)', 
              right: 'calc(-50vw + 50%)',
              width: '100vw'
            }} 
          />
          <div 
            className="absolute bottom-0 h-px bg-neutral-300 z-0"
            style={{ 
              left: 'calc(-50vw + 50%)', 
              right: 'calc(-50vw + 50%)',
              width: '100vw'
            }} 
          />
        </>
      )}
      
      <div
        className={cn(
          "relative border border-neutral-300 bg-white",
          connectTo === "top" && "-mt-px",
          connectTo === "bottom" && "-mb-px", 
          connectTo === "both" && "-my-px",
          noDecoration === "top" && "border-t-transparent",
          noDecoration === "bottom" && "border-b-transparent",
          noDecoration === "both" && "border-t-transparent border-b-transparent",
          className,
        )}
        id={id}
      >
        {minimal && noDecoration !== "both" && (
          <>
            {noDecoration !== "top" && (
              <>
                <div className="absolute -top-1 -left-1 w-2 h-2 rounded-full bg-neutral-400 opacity-60"></div>
                <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-neutral-400 opacity-60"></div>
              </>
            )}
            {noDecoration !== "bottom" && (
              <>
                <div className="absolute -bottom-1 -left-1 w-2 h-2 rounded-full bg-neutral-400 opacity-60"></div>
                <div className="absolute -bottom-1 -right-1 w-2 h-2 rounded-full bg-neutral-400 opacity-60"></div>
              </>
            )}
          </>
        )}
        
        <div
          className={cn(
            "grid gap-px bg-neutral-300",
            columns === 1 && "grid-cols-1",
            columns === 2 && "grid-cols-2",
            columns === 3 && "grid-cols-3", 
            columns === 4 && "grid-cols-4",
          )}
        >
          {children}
        </div>
      </div>
    </div>
  )
}

interface GridItemProps {
  children: React.ReactNode
  className?: string
  padding?: "none" | "sm" | "md" | "lg"
}

export function GridItem({ children, className, padding = "md" }: GridItemProps) {
  return (
    <div 
      className={cn(
        "bg-white text-neutral-900",
        padding === "none" && "p-0",
        padding === "sm" && "p-4",
        padding === "md" && "p-6",
        padding === "lg" && "p-8",
        className
      )}
    >
      {children}
    </div>
  )
}
