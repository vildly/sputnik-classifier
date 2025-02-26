import { useState } from "react"
import JSONViewer from "../components/JSONViewer"
import Error from "../components/Error"

export default function InspectView() {
    const [error, setError] = useState<string | null>(null)

    function handleViewerError(error: string | null): void {
        setError(error)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Error message={error} />}
            {!error && <JSONViewer
                data={`[{"hello":"world"}]`}
                errorCallback={handleViewerError}
                className="text-white"
            />}
        </div>
    )
}
