import type React from "react"

import { useState } from "react"
import { Plus, Trash2, AlertCircle } from "lucide-react"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"

// URL validation schema
const urlSchema = z.string().url("Please enter a valid URL")

export interface UrlManagerProps {
  url: string
  onUrlChange: (url: string) => void
}

export default function UrlManager({ url, onUrlChange }: UrlManagerProps) {
  const [inputValue, setInputValue] = useState("")
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const validateAndAddUrl = () => {
    if (!inputValue.trim()) return

    // Reset previous error
    setError(null)

    try {
      // Validate URL format
      urlSchema.parse(inputValue)

      // Set the URL
      onUrlChange(inputValue)
      setInputValue("")
      setOpen(false)
    } catch (err) {
      if (err instanceof z.ZodError) {
        setError(err.errors[0].message)
      } else {
        setError("An error occurred while validating the URL")
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      validateAndAddUrl()
    }
  }

  const handleClearUrl = () => {
    onUrlChange('')
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <Plus className="h-4 w-4" />
          <span className="sr-only">Manage URL</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>URL Manager</DialogTitle>
          {url && <p className="text-sm text-muted-foreground mt-1">当前URL: {url}</p>}
        </DialogHeader>
        <div className="flex items-center space-x-2">
          <Input
            placeholder="Enter URL (e.g., https://example.com)"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value)
              setError(null) // Clear error when input changes
            }}
            onKeyDown={handleKeyDown}
            className={`flex-1 ${error ? "border-destructive" : ""}`}
          />
          <Button size="icon" onClick={validateAndAddUrl}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        {error && (
          <Alert variant="destructive" className="mt-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {url && (
          <div className="mt-4">
            <div className="rounded-md border p-2 text-sm break-all flex items-center justify-between">
              <span className="mr-2">{url}</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleClearUrl}
                className="h-8 w-8 text-destructive hover:bg-destructive/10"
              >
                <Trash2 className="h-4 w-4" />
                <span className="sr-only">Clear</span>
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}