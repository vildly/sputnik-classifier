import { cn } from "../lib/utils"

interface BannerProps {
    text: string
    className?: string
}

export default function Banner({ text, className }: BannerProps) {
  return (
    <div
        className={cn(
            "max-w-2xl p-6",
            "bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500",
            "rounded-xl shadow-lg",
            "transform hover:scale-105 transition-all duration-700",
            className
        )}
    >
      <h1 className="text-black text-4xl font-extrabold">
        {text}
      </h1>
    </div>
  )
}
