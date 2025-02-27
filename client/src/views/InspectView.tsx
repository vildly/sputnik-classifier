import { useState } from "react"
import { getData } from "../services/api"
import Input from "../components/Input"
import Form from "../components/Form"
import Loading from "../components/Loading"
import { HexColors } from "../lib/colors"
import PreViewer from "../components/JSONViewer"
import Flash from "../components/Flash"

export default function InspectView() {
    const [error, setError] = useState<string>("")
    const [id, setId] = useState<string>("")
    const [data, setData] = useState<any>("")
    const [processing, setProcessing] = useState<boolean>(false)

    async function handleSubmit(): Promise<void> {
        try {
            // Prevent if no id was set
            if (!id) return
            // Prevent re-submission
            if (processing) return

            setProcessing(true)
            const res = await getData(id)
            if (!res || typeof res !== "object") {
                console.info(res)
                throw new Error("Could not parse the response")
            }
            setData(JSON.stringify(res, null, 2))

            // Reset
            setError("")
            setProcessing(false)
        } catch(err: any) {
            setError(err.message)
            setProcessing(false)
        }
    }

    function handleId(e: React.ChangeEvent<HTMLInputElement>): void {
        setId(e.target.value)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Flash message={error} className="bg-red-900 border-red-400" />}
            <Form onSubmit={handleSubmit} className="flex-row">
                <Input
                    type="text"
                    placeholder="Enter a job ID..."
                    onChange={handleId}
                    className="w-full bg-neutral-500 border-white border-2 hover:bg-neutral-600 focus:bg-neutral-600"
                />
                {!processing && id && <Input
                    type="submit"
                    value="Get"
                    className="cursor-pointer"
                />}
            </Form>
            {processing && <Loading color={HexColors.WHITE} className="mx-auto" />}
            {data && <PreViewer value={data} className="text-white" />}
        </div>
    )
}
