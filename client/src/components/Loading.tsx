import { HexColors } from "../lib/colors"
import { cn } from "../lib/utils"

interface LoadingProps {
    color?: HexColors
    size?: number
    className?: string
}

export default function Loading({ color = HexColors.BLACK, size = 24, className }: LoadingProps) {
    // Render an SVG spinner that rotates using Tailwind's animate-spin class
    return (
        <svg
            // Put the className prop before spin to ensure this property
            // cant be changed
            className={cn(className, "animate-spin")}
            // Sets the SVG's width and height based on the size prop
            style={{ width: size, height: size }}
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
        >
            {/*
                The circle element draws the static, faint background track of the spinner.
                - cx and cy define the center of the circle (12,12 in a 24x24 viewbox)
                - r sets the radius of the circle (10 units)
                - stroke uses the provided color for the circle outline
                - strokeWidth sets the thickness of the circle's outline
                - opacity-25 makes the circle semi-transparent
            */}
            <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke={color}
                strokeWidth="4"
            />
            {/*
                The path element draws the visible spinning segment of the loader.
                This segment is styled to be more opaque (opacity-75) and filled with the provided color.
                The 'd' attribute defines the geometric shape (a wedge) that forms the moving part of the spinner:
                - M4 12: Moves the "pen" to (4,12)
                - a8 8 0 018-8: Draws an arc with an 8-unit radius ending at (12,4)
                - V0: Draws a vertical line to y=0 at the current x-coordinate (12)
                - C5.373 0 0 5.373 0 12: Draws a cubic Bézier curve from (12,0) to (0,12) using control points
                - h4: Draws a horizontal line of 4 units to bring the path back to (4,12)
            */}
            <path
                className="opacity-75"
                fill={color}
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
        </svg>
    )
}
